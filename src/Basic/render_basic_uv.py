from pathlib import Path

import numpy as np
import slangpy as spy

# --- Configuration ---
DIM_X = 128
DIM_Y = DIM_X
DIR_CURRENT = Path(__file__).parent
SHADER_FILE_NAME = Path(__file__).stem + ".slang"
INPUT_IMAGE_FILE = DIR_CURRENT.parent / "assets" / "TX_Essentials_ColorsExample_B.png"
OUTPUT_IMAGE_FILE = DIR_CURRENT / "temp" / "output.png"

# --- Device Initialization ---
print("Creating Slang device...")
device = spy.Device(
    enable_debug_layers=True,
    compiler_options=spy.SlangCompilerOptions({"include_paths": [DIR_CURRENT]}),
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

# Save an image
if not OUTPUT_IMAGE_FILE.parent.is_dir():
    OUTPUT_IMAGE_FILE.parent.mkdir()

output_bmp = output_tex.to_bitmap().convert(
    pixel_format=spy.Bitmap.PixelFormat.rgb,
    component_type=spy.Bitmap.ComponentType.uint8,
    srgb_gamma=False,
)
output_bmp.write(OUTPUT_IMAGE_FILE)

print(f"File saved as {OUTPUT_IMAGE_FILE}.")

# Display in a running TEV instance (https://github.com/Tom94/tev)
# spy.tev.show(output_tex, name="Rendered Output")

# Display with matplotlib
import matplotlib.pyplot as plt
fig, plot_axis = plt.subplots(1, 1, figsize=(5,5), tight_layout=True)
plot_axis.imshow(output_bmp)
plot_axis.set_title("Output")
plot_axis.axis("off")
plt.show()
