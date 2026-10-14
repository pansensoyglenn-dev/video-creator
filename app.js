window.__appLoaded = true;
const $ = (id) => document.getElementById(id);


const py = (s) => JSON.stringify(s.replace(/\s+/g, " ").trim());

function clamp(value, min, max, fallback) {
  const n = parseInt(value, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

function safeFileName(name) {
  const cleaned = name.trim().replace(/[^\w.\-]/g, "_");
  if (!cleaned) return "output_6s.mp4";
  return cleaned.toLowerCase().endsWith(".mp4") ? cleaned : cleaned + ".mp4";
}


function buildStep1(v) {
  return `#@title 🎬 Step 1 — Make keyframes (SDXL)
!pip install -q diffusers transformers accelerate safetensors

#@markdown ### Prompts
prompt = ${py(v.prompt)} #@param {type:"string"}
negative_prompt = ${py(v.negative)} #@param {type:"string"}
SEED = ${v.seed} #@param {type:"integer"}

import os
import torch
from diffusers import StableDiffusionXLPipeline
from IPython.display import display

print("Loading SDXL...")
sdxl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    low_cpu_mem_usage=True,
)
sdxl.enable_model_cpu_offload()

for i in range(3):
    g = torch.Generator("cpu").manual_seed(SEED + i)
    img = sdxl(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1344, height=768,
        num_inference_steps=30,
        guidance_scale=6.5,
        generator=g,
    ).images[0]
    img.save(f"keyframe_{i}.png")
    print(f"Candidate {i}")
    display(img.resize((672, 384)))

print("Keyframes saved. Restarting the session to free RAM...")
print("A 'session crashed' message is NORMAL here. Now run Step 2.")
os.kill(os.getpid(), 9)
`;
}


function buildStep2(v) {
  return `#@title 🎬 Step 2 — Animate keyframe (SVD)
!pip install -q diffusers transformers accelerate safetensors

#@markdown ### Settings
SEED = ${v.seed} #@param {type:"integer"}
PICK = ${v.pick} #@param [0, 1, 2] {type:"raw"}
N_CLIPS = ${v.clips} #@param {type:"slider", min:1, max:4, step:1}
MOTION = ${v.motion} #@param {type:"slider", min:30, max:200, step:5}
OUTPUT_NAME = ${py(v.outname)} #@param {type:"string"}

import gc
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

keyframe = Image.open(f"keyframe_{PICK}.png").convert("RGB").resize((1024, 576))

print("Loading Stable Video Diffusion...")
svd = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16, variant="fp16",
    low_cpu_mem_usage=True,
)
svd.enable_model_cpu_offload()
svd.unet.enable_forward_chunking()

all_frames = []
current = keyframe
for n in range(N_CLIPS):
    print(f"Animating clip {n + 1}/{N_CLIPS}...")
    g = torch.manual_seed(SEED + n)
    out = svd(
        current,
        num_frames=25,
        decode_chunk_size=2,
        motion_bucket_id=MOTION,
        noise_aug_strength=0.02,
        fps=7,
        generator=g,
    ).frames[0]
    all_frames += out if n == 0 else out[1:]
    current = out[-1]

export_to_video(all_frames, "output_raw.mp4", fps=8)

del svd
gc.collect()
torch.cuda.empty_cache()

print("Post-processing with FFmpeg...")
!ffmpeg -y -loglevel error -i output_raw.mp4 \\
  -vf "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1,scale=1280:720:flags=lanczos,eq=contrast=1.05:saturation=1.08" \\
  -c:v libx264 -pix_fmt yuv420p -crf 18 {OUTPUT_NAME}

from google.colab import files
files.download(OUTPUT_NAME)
print("Done.")
`;
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  try {

  const values = {
    prompt: $("prompt").value,
    negative: $("negative").value,
    seed: clamp($("seed").value, 0, 2147483647, 42),
    pick: clamp($("pick").value, 0, 2, 1),
    clips: clamp($("clips").value, 1, 4, 2),
    motion: clamp($("motion").value, 30, 200, 90),
    outname: safeFileName($("outname").value),
  };

  if (!values.prompt.trim()) {
    $("prompt").focus();
    return;
  }

  $("code1").value = buildStep1(values);
  $("code2").value = buildStep2(values);
  $("result").hidden = false;
  $("result").scrollIntoView({ behavior: "smooth" });
  $("status").hidden = true;
  } catch (err) {
    $("status").textContent = "Error: " + err.message;
    $("status").hidden = false;
  }
});

document.querySelectorAll(".copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const box = $(btn.dataset.target);
    const label = btn.dataset.label;
    try {
      await navigator.clipboard.writeText(box.value);
    } catch (err) {
      box.focus();
      box.select();
      document.execCommand("copy");
    }
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = label), 1500);
  });
});

document.querySelectorAll(".clear").forEach((btn) => {
  btn.addEventListener("click", () => {
    const box = $(btn.dataset.clear);
    box.value = "";
    box.focus();
  });
});
