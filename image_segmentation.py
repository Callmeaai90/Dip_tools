import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Main Window ----------
root = Tk()
root.title("Basic Image Segmentation GUI")
root.state('zoomed')
root.configure(bg="#f0f0f0")

main_frame = Frame(root, bg="#f0f0f0")
main_frame.pack(fill="both", expand=True, padx=40, pady=40)

# ---------- Image Display ----------
plot_frame = Frame(main_frame)
plot_frame.pack(side="top", fill="both", expand=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # Original, Grayscale, Result
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True)

# ---------- State ----------
image_color = None
image_gray = None

# ---------- Segmentation Operations ----------
def canny_edge(img):
    return cv2.Canny(img, 100, 200)

def otsu_threshold(img):
    _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result

def watershed_segmentation(img):
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_color, markers)

    img_color[markers == -1] = [0, 255, 0]  # Green boundaries
    kernel_thick = np.ones((5, 5), np.uint8)
    mask = np.uint8(markers == -1)
    mask_dilated = cv2.dilate(mask, kernel_thick, iterations=2)
    img_color[mask_dilated == 1] = [0, 255, 0]

    return img_color

segmentation_ops = {
    "Canny Edge Detection": canny_edge,
    "Otsu Thresholding": otsu_threshold,
    "Watershed Algorithm": watershed_segmentation
}

# ---------- Process Image ----------
def apply_operation(op_name):
    if image_color is None or image_gray is None:
        info_label['text'] = "❌ Please select an image first."
        return

    processed = segmentation_ops[op_name](image_gray)

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    axes[0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=18)
    axes[1].imshow(image_gray, cmap='gray')
    axes[1].set_title("Grayscale Image", fontsize=18)
    if op_name == "Watershed Algorithm":
        axes[2].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        axes[2].imshow(processed, cmap='gray')
    axes[2].set_title(f"Segmented: {op_name}", fontsize=18)
    for ax in axes: ax.axis('off')
    canvas.draw()
    info_label['text'] = f"✅ Applied: {op_name}"

# ---------- Load Image ----------
def load_image():
    global image_color, image_gray
    file_path = filedialog.askopenfilename(title="Select an image")
    if not file_path:
        info_label['text'] = "❌ No image selected."
        return
    image_color = cv2.imread(file_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    info_label['text'] = f"✅ Image Loaded: {file_path.split('/')[-1]}"

# ---------- UI Layout ----------
info_label = Label(main_frame, text="🖼️ Select an image to begin", font=("Arial", 28, "bold"), bg="#f0f0f0")
info_label.pack(pady=20)

control_frame = Frame(main_frame, bg="#f0f0f0")
control_frame.pack(pady=10)

Button(control_frame, text="📁 Select Image", command=load_image,
       font=("Arial", 22, "bold"), bg="#3498db", fg="white", width=25, height=2).grid(row=0, column=0, padx=20, pady=10)

Button(control_frame, text="Canny Edge Detection", command=lambda: apply_operation("Canny Edge Detection"),
       font=("Arial", 22, "bold"), bg="#1abc9c", fg="white", width=25, height=2).grid(row=0, column=1, padx=20, pady=10)

Button(control_frame, text="Otsu Thresholding", command=lambda: apply_operation("Otsu Thresholding"),
       font=("Arial", 22, "bold"), bg="#e67e22", fg="white", width=25, height=2).grid(row=1, column=0, padx=20, pady=10)

Button(control_frame, text="Watershed Algorithm", command=lambda: apply_operation("Watershed Algorithm"),
       font=("Arial", 22, "bold"), bg="#9b59b6", fg="white", width=25, height=2).grid(row=1, column=1, padx=20, pady=10)


root.mainloop()
