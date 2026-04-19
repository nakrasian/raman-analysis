import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ANIMATE_FOLDER = "animations"
PLOT_FOLDER = "plots/raw"
METADATA_FOLDER = "metadata-selected"
SPECTRA_FOLDER = "spectra-selected"
SPECTRA_LENGTH = 1024

def animate_spectra(spectra_name, wavelengths, intensities, label=None):
    # Placeholder for animation code
    print(f"Animating {spectra_name} with wavelengths {wavelengths[:5]} ... and intensities shape {intensities.shape}")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_title(f"Spectra Animation: {label}")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Intensity")

    ax.set_xlim(wavelengths.min(), wavelengths.max())
    ax.set_ylim(0, 10000)
    cmap = plt.get_cmap('viridis')

    for i in range(len(intensities)):
        color = cmap(i / len(intensities))
        ax.plot(wavelengths, intensities[i], color=color, alpha=0.3)

    plot_file = os.path.join(PLOT_FOLDER, f"{spectra_name}_plot.png")
    os.mkdir(PLOT_FOLDER) if not os.path.exists(PLOT_FOLDER) else None
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")
    plt.close()



    ## Save Animations
    # def update(frame):
    #     color = cmap(frame / len(intensities))
    #     line.set_color(color)
    #     line.set_ydata(intensities[frame])
    #     return line,


    # num_frames = len(intensities)
    # frames_forward = np.arange(num_frames)
    # frames_backward = frames_forward[::-1]
    # frames_bidirectional = np.concatenate([frames_forward, frames_backward])

    # ani = animation.FuncAnimation(fig = fig, func = update, frames = frames_bidirectional, interval=30)
    # # plt.show()
    # # plt.close()

    # animate_file = os.path.join(ANIMATE_FOLDER, f"{spectra_name}_animation.gif")
    # ani.save(animate_file, writer='pillow')
    # print(f"Saved animation to {animate_file}")

def read_spectra(spectra_path, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = f.read()


    # Extracting metadata information
    num_spectra = int(metadata.splitlines()[0].split(":")[1].strip())
    label = metadata.splitlines()[6].split(":")[1].strip()
    group = metadata.splitlines()[7].split(":")[1].strip()

    spectra = pd.read_csv(spectra_path, header=None, names=["Wavenumber", "Intensity"])
    wavelengths = spectra["Wavenumber"].head(SPECTRA_LENGTH).values
    intensities = spectra["Intensity"].values.reshape(num_spectra, SPECTRA_LENGTH)

    print (f"Number of spectra: {num_spectra}")
    print (f"Label: {label}")
    print (f"Group: {group}")
    print (f"Wavelengths: {wavelengths[:5]} ...")
    print (f"Intensities shape: {intensities.shape}")

    animate_spectra(os.path.basename(spectra_path), wavelengths, intensities, label=label)



def main():
    parser = argparse.ArgumentParser(description="Animate spectra and metadata")
    parser.add_argument("--spectra", default=".", help="Spectra number to animate (default: all)")
    args = parser.parse_args()

    if args.spectra == ".":
        spectra_numbers = range(1, 68)
    else:
        spectra_numbers = [int(num) for num in args.spectra.split(",")]
    for num in spectra_numbers:
        metadata_file = os.path.join(METADATA_FOLDER, f"Captured_spectra_{num}_metadata.txt")
        spectra_file = os.path.join(SPECTRA_FOLDER, f"Captured_spectra_{num}.txt")
        if os.path.exists(metadata_file) and os.path.exists(spectra_file):
            read_spectra(spectra_file, metadata_file)

if __name__ == "__main__":
    main()





