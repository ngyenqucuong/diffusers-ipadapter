from fastapi import FastAPI, File, UploadFile, Form, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import  AutoPipelineForText2Image,DDIMScheduler
from transformers import CLIPVisionModelWithProjection


from PIL import Image ,ImageDraw
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

from hidiffusion import apply_hidiffusion, remove_hidiffusion
from huggingface_hub import snapshot_download

from insightface.app import FaceAnalysis
from insightface.utils import face_align

import cv2

snapshot_download(
    repo_id="InstantX/InstantID", allow_patterns="/models/antelopev2/*", local_dir="./models/antelopev2/"
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for pipelines
pipe = None
executor = ThreadPoolExecutor(max_workers=1)
processor = None

face_analysis_app = None


def initialize_pipelines():
    """Initialize the diffusion pipelines with InstantID and SDXL-Lightning - GPU optimized"""
    global pipe, face_analysis_app
    
    try:
        # Clear CUDA cache before initialization
       
        logger.info("Loading face analysis model...")
        face_analysis_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
        ).to("cuda")

        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            image_encoder=image_encoder,
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        
        # pipe.load_ip_adapter(
        #     [ "h94/IP-Adapter", "h94/IP-Adapter-FaceID"],
        #     subfolder=[ "sdxl_models",""],
        #     weight_name=[
        #         "ip-adapter_sdxl_vit-h.safetensors",
        #         "ip-adapter-faceid-plusv2_sdxl.bin"
        #     ],
        #     image_encoder_folder=None,
        # )
        pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plusv2_sdxl.bin")
        pipe.enable_model_cpu_offload()
        # apply_hidiffusion(pipe)   
        
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
    ref_images_embeds = []
    ip_adapter_images = []
    adapter_weight_lst = [request.ip_adapter_scale]

    pipe.set_ip_adapter_scale(adapter_weight_lst)
    cv2_face_image = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    faces = face_analysis_app.get(cv2_face_image)
    facealign = face_align.norm_crop(cv2_face_image, landmark=faces[0].kps, image_size=224)
    ip_adapter_images.append(facealign)
    faceimage = torch.from_numpy(faces[0].normed_embedding)
    ref_images_embeds.append(faceimage.unsqueeze(0))
    ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
    neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
    id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=torch.float16, device="cuda")
    clip_embeds = pipe.prepare_ip_adapter_image_embeds([ip_adapter_images], None, torch.device("cuda"), 1, True)[0]
    seed = request.seed 
    if not request.seed:
        seed = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()

    pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
    pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False
    generated_image = pipe(
        ip_adapter_image_embeds=[id_embeds],
        prompt=request.prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=request.num_inference_steps,
        generator = torch.Generator(device="cuda").manual_seed(seed),
        num_images_per_prompt=1,

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
    num_inference_steps: int = Form(50),  # Number of inference steps
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