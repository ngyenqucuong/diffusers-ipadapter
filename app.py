from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from pydantic import BaseModel
from typing import Optional
import torch

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers.models import ControlNetModel, UNet2DConditionModel
from diffusers import EulerDiscreteScheduler
from PIL import Image 
import numpy as np
import io
import json
import uuid
import os
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from contextlib import asynccontextmanager
from transformers import CLIPVisionModelWithProjection

from safetensors.torch import load_file

from huggingface_hub import snapshot_download,hf_hub_download

from insightface.app import FaceAnalysis

import cv2

if not os.path.exists("./models/antelopev2/"):
    snapshot_download(
        repo_id="InstantX/InstantID", allow_patterns="/models/antelopev2/*", local_dir="./models/antelopev2/"
    )
    # run 'mv models/antelopev2/antelopev2/* models/antelopev2/' cmd
    os.system("mv ./models/antelopev2/antelopev2/* ./models/antelopev2/")

if not os.path.exists("./checkpoints/"):
    # make dir 'ControlNetModel'
    os.makedirs("./checkpoints/ControlNetModel", exist_ok=True)
    hf_hub_download(
        repo_id="InstantX/InstantID",
        subfolder="ControlNetModel",
        filename="config.json",
        local_dir="./checkpoints/ControlNetModel"
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        subfolder="ControlNetModel",
        filename="diffusion_pytorch_model.safetensors",
        local_dir="./checkpoints/ControlNetModel"
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="./checkpoints"
    )


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for pipelines
pipe = None
executor = ThreadPoolExecutor(max_workers=1)
processor = None

face_analysis_app = None

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image

def initialize_pipelines():
    """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
    global pipe, face_analysis_app
    
    try:
        # Clear CUDA cache before initialization
       
        logger.info("Loading face analysis model...")
        face_analysis_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Path to InstantID models
        face_adapter = './checkpoints/ip-adapter.bin'
        controlnet_path = './checkpoints/ControlNetModel'
        
        # Load ControlNet
        logger.info("Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        
        # SDXL-Lightning LoRA path
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_8step_unet.safetensors"
        
        # Base model path
        base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
        
        logger.info("Loading SDXL base pipeline...")
        unet = UNet2DConditionModel.from_config(base_model_path, subfolder="unet").to("cuda", torch.float16)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            unet=unet
        )
        pipe.cuda()
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
        pipe.load_ip_adapter_instantid(face_adapter)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

        
        
    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipelines on startup"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, initialize_pipelines)
    yield


app = FastAPI(title="SDXL Face Swap API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)







class Img2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    seed: Optional[int] = None
    strength: float = 0.8
    ip_adapter_scale: float = 0.8  # Lower for InstantID
    controlnet_conditioning_scale: float = 0.8
    guidance_scale: float = 0.0  # Zero for LCM
    detail_face: bool = False  # Whether to refine face details
    num_inference_steps: int = 50  # Number of inference steps

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None

# In-memory job storage
jobs = {}
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)



async def gen_img2img(job_id: str, face_image : Image.Image,pose_image: Image.Image,request: Img2ImgRequest):
    negative_prompt = f"{request.negative_prompt},monochrome, lowres, bad anatomy, worst quality, low quality"
    # pipe.set_ip_adapter_scale([request.strength,request.ip_adapter_scale])

    cv2_face_image = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    pose_image_cv2 =  cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR)


    faces = face_analysis_app.get(cv2_face_image)
    face_info = max(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    face_emb = face_info["embedding"]
    
    pose_faces = face_analysis_app.get(pose_image_cv2)
    pose_info = max(pose_faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
    pose_kps = draw_kps(pose_image, pose_info["kps"])

    width, height = pose_kps.size
    control_mask = np.zeros([height, width, 3])
    x1, y1, x2, y2 = face_info["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    control_mask[y1:y2, x1:x2] = 255
    control_mask = Image.fromarray(control_mask.astype(np.uint8))
    seed = request.seed
    if not request.seed:
        seed = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()
    generator = torch.Generator(device='cuda').manual_seed(seed)
    generated_image = pipe(
        prompt=request.prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=pose_kps,
        control_mask=control_mask,
        controlnet_conditioning_scale=request.controlnet_conditioning_scale,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=0,
        height=height,
        width=width,
        generator=generator,
        num_images_per_prompt=1
    ).images[0]
    
    filename = f"{job_id}_base.png"
    filepath = os.path.join(results_dir, filename)
    generated_image.save(filepath)
        
    metadata = {
        "job_id": job_id,
        "type": "head_swap",
        "seed": seed,
        "prompt": request.prompt,
        "parameters": request.dict(),
        "filename": filename,
        "device_used": 'cuda',
    }
        
    metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["progress"] = 1.0
    jobs[job_id]["result_url"] = f"/results/{filename}"
    jobs[job_id]["metadata"] = metadata
    jobs[job_id]["completed_at"] = datetime.now()
    
    logger.info(f"Img2img completed successfully on cuda")








@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    try:
        with open("interface.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Web interface not found</h1>")

@app.get("/web", response_class=HTMLResponse)
async def serve_web_interface_alt():
    """Alternative route for web interface"""
    return await serve_web_interface()

@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(),
            "gpu_memory_cached": torch.cuda.memory_reserved()
        }
    
    pipeline_device = None
    if pipe is not None:
        try:
            # Try to get device from unet (most reliable)
            pipeline_device = str(pipe.unet.device)
        except:
            try:
                # Fallback to vae device
                pipeline_device = str(pipe.vae.device)
            except:
                pipeline_device = "unknown"
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "pipeline_device": pipeline_device,
        "pipelines_loaded": pipe is not None,
        "gpu_info": gpu_info
    }


@app.post("/img2img")
async def img2img(
    base_image: UploadFile = File(...),
    pose_image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form("(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"),
    strength: float = Form(0.85),
    ip_adapter_scale: float = Form(0.8),  # Lower for InstantID
    controlnet_conditioning_scale: float = Form(0.8),
    num_inference_steps: int = Form(8),  # Number of inference steps
    detail_face: bool = Form(False),
    guidance_scale: float = Form(0),  # Zero for LCM
    seed: Optional[int] = Form(None),
    
):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "head_swap"
    }
    try:
    # Load images
        base_img = Image.open(io.BytesIO(await base_image.read())).resize((256, 256))
        pose_img = Image.open(io.BytesIO(await pose_image.read())).resize((512, 768))
        request = Img2ImgRequest(

            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            detail_face=detail_face,
            num_inference_steps=num_inference_steps
            
        )
        # Start background task
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, lambda: asyncio.run(
            gen_img2img(job_id, base_img, pose_img, request)
        ))
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        logger.error(f"Error processing img2img request: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        return {"job_id": job_id, "status": "failed", "error_message": str(e)}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result_url": job.get("result_url"),
        "seed": job.get("metadata", {}).get("seed"),
        "error_message": job.get("error_message"),
        "created_at": job["created_at"].isoformat(),
        "completed_at": job.get("completed_at").isoformat() if job.get("completed_at") else None
    }

@app.get("/results/{filename}")
async def get_result(filename: str):
    """Get result image"""
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath)


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    try:
        job_list = []
        for job_id, job_data in jobs.items():
            job_list.append({
                "job_id": job_id,
                "status": job_data.get("status", "unknown"),
                "created_at": job_data.get("created_at", datetime.now()).isoformat(),
                "completed_at": job_data.get("completed_at").isoformat() if job_data.get("completed_at") else None,
                "result_url": job_data.get("result_url"),
                "error_message": job_data.get("error_message")
            })
        
        job_list.sort(key=lambda x: x["created_at"], reverse=True)
        return job_list
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return []

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete files
    job = jobs[job_id]
    if "metadata" in job and "filename" in job["metadata"]:
        filename = job["metadata"]["filename"]
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete metadata file
        metadata_path = os.path.join(results_dir, f"{job_id}_metadata.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    # Remove from jobs
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    
    # Set environment variables for better CUDA error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    uvicorn.run(app, host="0.0.0.0", port=8888)