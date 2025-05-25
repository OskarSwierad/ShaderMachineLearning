import slangpy as spy
import numpy as np
from pathlib import Path

# --- Configuration ---
DIM_X = 128
DIM_Y = DIM_X
DIR_CURRENT = Path(__file__).parent
SHADER_FILE_NAME = Path(__file__).stem + ".slang"
INPUT_IMAGE_FILE = DIR_CURRENT.parent / "assets" / "TX_Essentials_ColorsExample_B.png"
OUTPUT_IMAGE_FILE = DIR_CURRENT / "temp" / "output.png"
TEV_PATH = "tev"  # Or a full path to TEV if not in PATH

# --- Device Initialization ---
print("Creating Slang device...")
device = spy.Device(
    enable_debug_layers=True,
    compiler_options={"include_paths": [DIR_CURRENT]},
    enable_print=True,
)
print("Slang device created.")

# --- Create Output Texture ---
print("Creating output texture...")
output_tex = device.create_texture(
    format=spy.Format.rgba32_float,
    width=DIM_X,
    height=DIM_Y,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    label="my_render_texture",
    data=np.ones((DIM_X, DIM_Y, 4), dtype=np.float32)
)
print("Output texture ready.")

# --- Load Shader ---
print(f"Loading shader program from '{SHADER_FILE_NAME}'...")
program = device.load_program(SHADER_FILE_NAME, ["MainCS"])
kernel = device.create_compute_kernel(program)
print("Shader program loaded.")

# --- Dispatch Shader ---
print("Dispatching compute shader...")
command_encoder = device.create_command_encoder()
kernel.dispatch(
    thread_count=[DIM_X, DIM_Y, 1],
    vars={
        "Params": {
            "Dimensions": spy.uint2(output_tex.width, output_tex.height),
            "OutTexture": output_tex
        }
    },
    command_encoder=command_encoder
)
device.submit_command_buffer(command_encoder.finish())
print("Shader dispatched.")

device.flush_print()

# --- Save and View Output Image ---
print("Saving and displaying output image...")

# Display in TEV
spy.tev.show(output_tex, name="Rendered Output")

# Save an image
if not OUTPUT_IMAGE_FILE.parent.is_dir():
    OUTPUT_IMAGE_FILE.parent.mkdir()

output_tex.to_bitmap().convert(
    pixel_format=spy.Bitmap.PixelFormat.rgb,
    component_type=spy.Bitmap.ComponentType.uint8,
    srgb_gamma=False,
).write(OUTPUT_IMAGE_FILE)

print(f"File saved as {OUTPUT_IMAGE_FILE}.")
