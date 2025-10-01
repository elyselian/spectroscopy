import spe_loader as sl

iccd_images = {}         


def load_images(image_paths):
    for path in image_paths:
        image_id = path.split("/")[-1].split(".")[0]
        
        spe_image = spe2py.load(path)
        
        iccd_images[image_id] = {
            "chords": {
                f"chord_{i}": {
                    "wavelength_axis": spe_image.wavelength_axis[i-1],  # Example: replace with actual method to access data for this chord
                    "pixel_axis": spe_image.pixel_axis[i-1]             # Example: replace with actual method to access data for this chord
                }
                for i in range(1, 21)
            },
            "V_c": None,  # Replace with actual method to access data
            "V_a": None, # Replace with actual method to access data
            "iccd_delay": None,           # Replace with actual method to access data
            "nosecone_id": None,      # Replace with actual method to access data
            "center_wavelength": None,    # Replace with actual method to access data
            "exposure": None,             # Replace with actual method to access data
            "telescope": None,            # Replace with actual method to access data
            "location": None,            # Replace with actual method to access data
        }

image_paths = ["path/to/image1.spe", "path/to/image2.spe"]
load_images(image_paths)

import pickle
import os

output_file = "iccd_images.pkl"
with open(output_file, "wb") as f:
    pickle.dump(iccd_images, f)
