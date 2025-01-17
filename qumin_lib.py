
import numpy as np
from matplotlib import pyplot as plt
import concurrent.futures
from PIL import Image
from matplotlib.widgets import Slider

## original functions

def Propagation_RS_zero_padd(tim, lamb, z, n, pp, zero_padd, w):
    """
    Simulates free space propagation of coherent light from plane 1 (transmittance 'tim')
    to a plane 2 distant 'z' in Rayleigh-Sommerfeld approximation.

    Parameters:
        tim : 2D array
            Transmittance of the plane
        lamb : float
            Wavelength
        z : float
            Propagation distance along the optical axis
        n : float
            Refractive index of the medium
        pp : float
            Pixel pitch
        zero_padd : int
            Zero padding option
        w : int
            Warning level (0 = no warnings, 1 = show warnings)

    Returns:
        Ap : 2D array
            Output plane after propagation
    """
    L, C = tim.shape
    if L != C and w == 1:
        print("Columns number is not equal to rows number.")
        print("Warning: The number of rows is used to select the method 'impulse response' or 'transfer function'.")

    # Test if the maximum instantaneous frequency satisfies NS
    finst_max_pp = (1 / lamb) * (L / 2 * pp) / np.sqrt((L / 2 * pp)**2 + z**2) * pp  # in pixels

    if zero_padd == 0:
        if finst_max_pp < 0.5:
            if w == 1:
                print("The RS kernel is computed in space domain")
            hRSz = RayleighSommerfeldFunction(L, C, z, n, lamb, pp, w)
            HRSz = np.fft.fft2(np.fft.ifftshift(hRSz))  # NumPy coordinate origin
            if w == 2:
                plt.figure("hRSz")
                plt.imshow(np.real(hRSz), cmap='gray')
                plt.colorbar()
                plt.axis('image')
                plt.title('real(hRSz)')
                plt.show()
            if w >= 1:
                print("Use of the impulse response for the propagation computation")
            fftt = np.fft.fft2(tim)  # NumPy coordinate origin
            Ap = np.fft.ifft2(HRSz * fftt)
        else:
            if w == 1:
                print("The RS kernel is computed in frequency domain")
            HRSz = RayleighSommerfeldFunctionFT(L, C, z, n, lamb, pp, w)  # NumPy coordinate origin
            if w == 2:
                plt.figure("HRSz")
                plt.imshow(np.real(HRSz), cmap='gray')
                plt.colorbar()
                plt.axis('image')
                plt.title('real(HRSz)')
                plt.show()
            if w >= 1:
                print("Use of the transfer function for the propagation computation")
            fftt = np.fft.fft2(tim)  # NumPy coordinate origin
            Ap = np.fft.ifft2(np.fft.ifftshift(HRSz) * fftt)

    else:  # (zero_padd >= 1)
        print("Zero padding not implemented yet")

    return Ap







def RayleighSommerfeldFunction(L, C, z, n, lambda_, pp, w):
    """
    Rayleigh Sommerfeld kernel
    Input parameters:
    L, C : number of lines and columns of the Fresnel function (output parameter)
    z : distance of propagation in meter
    n : refractive index of the medium propagation
    lambda_ : wavelength in meter
    pp : pixel pitch in meter
    w : parameter for sampling check
    Output:
    hRSz : matrix of L lines and C columns of complex values of the Rayleigh-Sommerfeld kernel
    """
    lambda_n = lambda_ / n

    if w == 1:
        finst_max_pp = (1 / lambda_n) * (L / 2 * pp) / np.sqrt((L / 2 * pp) ** 2 + z ** 2) * pp
        if finst_max_pp <= 0.5:
            print("bon échantillonnage de la fonction de RS")
        else:
            print("mauvais échantillonnage de la fonction de RS")

    hRSz = np.zeros((L, C), dtype=complex)
    Ki, Et = np.meshgrid(np.arange(-np.floor(C / 2), np.ceil(C / 2)),
                         np.arange(-np.floor(L / 2), np.ceil(L / 2)))

    R2 = (Ki * pp) ** 2 + (Et * pp) ** 2 + z ** 2
    R = np.sign(z) * np.sqrt(R2)
    k = 2 * np.pi / lambda_n
    hRSz = (pp ** 2) * (z / (1j * lambda_n * R ** 2)) * np.exp(1j * k * R)

    return hRSz





def RayleighSommerfeldFunctionFT(L, C, z, n, lambda_, pp, w):
    """
    Rayleigh-Sommerfeld kernel computed in Fourier Domain
    Input parameters:
    L, C: number of lines and columns of the Fresnel function (output parameter)
    z: distance of propagation in meters
    n: refractive index of the medium propagation
    lambda_: wavelength in meters
    pp: pixel pitch in meters
    w: flag for sampling check
    Output parameter:
    HRSz: matrix of L lines and C columns of complex values of the
    Fourier Transform of Rayleigh-Sommerfeld kernel
    """

    lambda_n = lambda_ / n

    if w == 1:
        finst_max_pp = (1 / lambda_n) * (L / 2 * pp) / np.sqrt((L / 2 * pp)**2 + z**2) * pp
        if finst_max_pp >= 0.5:
            print('bon échantillonnage de la fonction de RS')
        else:
            print('mauvais échantillonnage de la fonction de RS')

    HRSz = np.zeros((L, C), dtype=np.complex_)
    mux, muy = np.meshgrid(
        np.arange(-np.floor(C / 2), np.ceil(C / 2)) / C,
        np.arange(-np.floor(L / 2), np.ceil(L / 2)) / L
    )
    mux_lbda = mux / pp * lambda_n
    muy_lbda = muy / pp * lambda_n

    arg2 = 1 - mux_lbda**2 - muy_lbda**2
    ind = np.where(arg2 > 0)
    sqrt_arg = np.zeros((L, C))
    sqrt_arg[ind] = np.sqrt(arg2[ind])

    k = 2 * np.pi / lambda_n
    HRSz[ind] = np.exp(1j * k * z * sqrt_arg[ind])

    return HRSz



## encapsulation

def backpropag_encapsulated(sqrt_im_norm_holo, lambda_, z_val, n, pp, zero_padd, w):
    """
    Encapsulates all the computation in one single function that does all the work
    """
    # compute the backpropagation
    bckprpg = Propagation_RS_zero_padd(sqrt_im_norm_holo, lambda_, -z_val, n, pp, zero_padd, w)
    bckprpg_norm = bckprpg / np.median(bckprpg)

    # extract intensity and phase
    bck_I = np.abs(bckprpg_norm)
    bck_phi = np.angle(bckprpg_norm)

    # already in uint8
    bck_I = (128 * bck_I).astype(np.uint8)
    bck_phi = (128 + 128 * bck_phi / np.pi * 2).astype(np.uint8)

    return(bck_I, bck_phi)


## ploting function

def backpro_live_plot(figure,bck_I,bck_phi):
        """
        just a plotting function usefull for cleaning main program
        """
        figure.clear()
        figure.add_subplot(1, 2, 1)
        plt.imshow(bck_I, aspect='equal', cmap='gray')
        plt.colorbar()
        plt.clim(0, 256)
        plt.title("intensity")

        plt.subplot(1, 2, 2)
        plt.imshow(bck_phi, aspect='equal', cmap='gray')
        plt.colorbar()
        plt.clim(0,256)
        plt.title("phase")
        plt.pause(0.05)
        return(None)






def display_tiff(file_path,title = "images"):
    # Open the multi-page TIFF file
    img = Image.open(file_path)

    # Get the number of pages in the TIFF
    num_pages = img.n_frames

    # Pre-convert all pages to NumPy arrays and store them in a list
    pages = []
    for i in range(num_pages):
        img.seek(i)
        pages.append(np.array(img))

    # Function to update the image when the slider changes
    def update(v):
        page = int(slider.val)  # Get the page number from the slider
        img_array = pages[page-1]  # Get the pre-converted NumPy array for the current page
        ax.imshow(img_array, cmap='gray')  # Update the displayed image
        plt.draw()  # Redraw the figure to update the display

    # Function to navigate images using keyboard arrows
    def on_key(event):
        page = int(slider.val)
        if event.key == 'right' and page < num_pages:  # Right arrow key
            page += 1
        elif event.key == 'left' and page > 1:  # Left arrow key
            page -= 1
        slider.set_val(page)  # Update the slider position
        update(None)  # Update the image based on new slider value

    # Create the figure and axis for displaying the image
    fig, ax = plt.subplots()

    # Initial image display
    page = 1
    img_array = pages[page-1]  # Get the first page as a NumPy array
    ax.imshow(img_array, cmap='gray')  # Display the first page
    ax.set_title(title)
    ax.axis('off')

    # Add a slider for page navigation
    ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Page', 1, num_pages, valinit=1, valstep=1)

    # Connect the slider to the update function
    slider.on_changed(update)

    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()












## Image saving optimisation


def save_as_multipage_tiff(images, output_path):
    """
    Save a list of grayscale NumPy arrays as a multipage TIFF file.

    Parameters:
        images (list of numpy.ndarray): List of grayscale images to save.
        output_path (str): Path to the output TIFF file.
    """

    # Convert NumPy arrays to PIL Image objects in 'L' mode (grayscale)
    pil_images = [Image.fromarray(img).convert("L") for img in images]

    # Save as multipage TIFF
    pil_images[0].save(
        output_path, save_all=True, append_images=pil_images[1:], format="TIFF"
    )



## Multithreading


def process_backpropagation(i, z_val, sqrt_im_norm_holo, lambda_, n, pp, zero_padd, w):
    """
    multithreaded call version of the backpropag_encapsulated function
    """
    bck_I, bck_phi = backpropag_encapsulated(sqrt_im_norm_holo, lambda_, -z_val, n, pp, zero_padd, w)
    return i, bck_I, bck_phi

def parallel_backpropagation(z, sqrt_im_norm_holo, lambda_, n, pp, zero_padd, w):
    """
    Parallelize the backpropag_encapsulated function
    """
    phase_ims = [None] * len(z)
    intensity_ims = [None] * len(z)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, z_val in enumerate(z):
            futures.append(executor.submit(process_backpropagation, i, z_val, sqrt_im_norm_holo, lambda_, n, pp, zero_padd, w))

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            i, bck_I, bck_phi = future.result()
            phase_ims[i] = bck_phi
            intensity_ims[i] = bck_I

    return phase_ims, intensity_ims






