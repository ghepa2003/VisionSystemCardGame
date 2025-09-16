import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate
import os

from PIL import Image

from typing import *

# ------ Functions from previous labs ------
def extract_sift_features(image: np.ndarray) -> Tuple[Sequence[cv2.KeyPoint], cv2.UMat]:
    """
    Extracts SIFT descriptors from an image.

    Parameters:
    -----------
    image (numpy.ndarray): The input image.

    Returns:
    --------
    Tuple[Sequence[cv2.KeyPoint], cv2.UMat]: A tuple containing the keypoints and descriptors.
    """
    
    # Create a SIFT detector
    sift = cv2.SIFT.create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

def match_descriptors(
        descriptors_src: cv2.UMat, 
        descriptors_tgt: cv2.UMat, 
        max_ratio: float = 0.7) -> Tuple[Sequence[Sequence[cv2.DMatch]], Sequence[Sequence[cv2.DMatch]]]:
        
    """
    Matches descriptors from the test image to the training descriptors using brute-force
    matches and Lowe's ratio test.

    Returns the good matches and all matches.

    Parameters:
    -----------
    descriptors_src (cv2.UMat): The source image descriptors.
    descriptors_tgt (cv2.UMat): The target image descriptors.
    max_ratio (float): The maximum ratio for Lowe's ratio test.

    Returns:
    --------
    Tuple[Sequence[Sequence[cv2.DMatch]], Sequence[Sequence[cv2.DMatch]]]: A tuple containing the good matches and all matches.

    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors_src, descriptors_tgt, k=2)

    # Apply Lowe's ratio test
    good_matches = []

    # Iterate through the matches and apply the ratio test
    # selecting only the good matches
    for i, (m,n) in enumerate(matches):
        if m.distance < max_ratio * n.distance:
            good_matches.append([m])

    return good_matches, matches

def augment_data(examples_train, labels_train, M):
    """
    Data augmentation: Takes each sample of the original training data and 
    applies M random rotations, which result in M new examples.
    
    Parameters:
    - examples_train: List of training examples (each example is a 2D array)
    - labels_train: Array of labels corresponding to the examples
    - M: Number of random rotations to apply to each training example
    
    Returns:
    - examples_train_aug: Augmented examples after rotations
    - labels_train_aug: Corresponding labels for augmented examples
    """
    # Initialize lists to store augmented data and labels
    examples_train_aug = []
    labels_train_aug = []
    
    # Loop through each training example
    for i in range(len(examples_train)):
        example = examples_train[i]
        label = labels_train[i]
        
        # Add the original example and label
        examples_train_aug.append(example)
        labels_train_aug.append(label)
        
        # Apply M random rotations
        for _ in range(M):
            angle = np.random.uniform(0, 360)  # Random rotation angle
            rotated_example = rotate(example, angle, reshape=False, mode='nearest')
            examples_train_aug.append(rotated_example)
            labels_train_aug.append(label)
    
    # Convert lists to numpy arrays
    examples_train_aug = np.array(examples_train_aug)
    labels_train_aug = np.array(labels_train_aug)
    
    return examples_train_aug, labels_train_aug

def residual_lgths(A, t, pts, pts_tilde):

    transformed_pts = A @ pts + t

    residuals = pts_tilde - transformed_pts

    lengths = np.sum(np.pow(residuals,2), axis=0)

    return lengths

def estimate_affine(pts, pts_tilde):
    
    num_points = pts.shape[1]

    A = np.zeros((num_points, 3), dtype=np.float64)
    A[:, 0] = pts[0, :] 
    A[:, 1] = pts[1, :] 
    A[:, 2] = 1           

    T1 = pts_tilde[0, :] 
    T2 = pts_tilde[1, :] 

    params1, residuals1, rank1, s1 = np.linalg.lstsq(A, T1, rcond=None)
    a, b, tx = params1

    params2, residuals2, rank2, s2 = np.linalg.lstsq(A, T2, rcond=None)
    c, d, ty = params2


    affine_matrix = np.array([[a, b, tx],
                              [c, d, ty]], dtype=np.float64)

    A = affine_matrix[:2, :2]  
    t = affine_matrix[:, 2].reshape(2,1)   

    return A, t

def ransac_fit_affine(pts: np.ndarray, pts_tilde: np.ndarray, threshold: float, n_iter: int = 10000, max_inliers: int = 0) -> Tuple[np.ndarray, np.ndarray]: 
    best_A = None
    best_t = None
    best_inlier_count = 0

    K = pts.shape[1]  

    for _ in range(n_iter):
        idx = np.random.choice(K, 3, replace=False)
        sample_pts = pts[:, idx]
        sample_tilde = pts_tilde[:, idx]

        try:
            A_temp, t_temp = estimate_affine(sample_pts, sample_tilde)
        except np.linalg.LinAlgError:
            continue  

        residuals = residual_lgths(A_temp, t_temp, pts, pts_tilde)
 
        inliers = residuals < threshold
        inlier_count = np.sum(inliers)

        
        if inlier_count > best_inlier_count and inlier_count > max_inliers:
            best_inlier_count = inlier_count
            best_A, best_t = estimate_affine(pts[:, inliers], pts_tilde[:, inliers])
            A=best_A
            t=best_t
            
    return A, t


# ------ Functions implemented for the system ------
def find_cards_in_image_blob(image_path, expected_num_cards):
    """
    Detects cards in an image using blob detection.
    Returns the image for drawing, the detected card contours (approx_poly), 
    and the original BGR image.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Impossibile load the image from'{image_path}'")
        return None, [], None 
    

    image_for_drawing = image_bgr.copy() 
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gaussian_kernel_size = (5, 5) 
    image_blurred = cv2.GaussianBlur(image_gray, gaussian_kernel_size, 0)

    adaptive_block_size = 45  
    adaptive_c_value = 20   
    
    
    thresh_img = cv2.adaptiveThreshold(
        image_blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        adaptive_block_size, 
        adaptive_c_value   
    )
   
    stamp_intermediate = False
    if stamp_intermediate:
        plt.imshow(thresh_img, cmap='gray'); plt.title("Thresholded (pre-morphology)"); plt.show()

    kernel_close_size = (9, 9) 
    kernel_close = np.ones(kernel_close_size, np.uint8)
    iterations_close = 2 

    morphed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel_close, iterations=iterations_close)

   
    if stamp_intermediate:
        plt.imshow(morphed_img, cmap='gray'); plt.title("Morphed (post-morphology)"); plt.show()

    contours, _ = cv2.findContours(
        morphed_img, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )

# --- Gemini's code for filtering contours based on area and aspect ratio ---
    candidate_cards = []
    
    epsilon_factor = 0.03 

    avg_expected_card_area = (image_bgr.shape[0] * image_bgr.shape[1]) / max(1, expected_num_cards)
    min_area_threshold = avg_expected_card_area * 0.20                                        
    max_area_threshold = avg_expected_card_area * 2.5  

    aspect_ratio_min_filter = 0.70
    aspect_ratio_max_filter = 1.30
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue 
        
        epsilon = epsilon_factor * perimeter 
        approx_poly = cv2.approxPolyDP(contour, epsilon, True)
        
        area = cv2.contourArea(approx_poly)
        x, y, w, h = cv2.boundingRect(approx_poly)
        aspect_ratio = float(w) / h if h != 0 else 0

        if len(approx_poly) == 4:
            if area > min_area_threshold and area < max_area_threshold:
                if aspect_ratio_min_filter < aspect_ratio < aspect_ratio_max_filter: 
                    candidate_cards.append(approx_poly)


    found_cards_contours = [] 
    if len(candidate_cards) >= expected_num_cards and expected_num_cards > 0:
        candidate_cards.sort(key=cv2.contourArea, reverse=True)
        found_cards_contours = candidate_cards[:expected_num_cards]
    
    else:
        found_cards_contours = candidate_cards
    
    return image_for_drawing, found_cards_contours, image_bgr
# ------ End of Gemini's code  ------




def load_sift_references(dataset_path, target_size, augmentation_m):
    """
    Loading SIFT reference images from a dataset folder.
    Each subfolder is considered a different card type.
    Each image in the subfolder is resized to target_size and augmented with M rotations.
    Returns a dictionary with card type as key and a list of (keypoints, descriptors) tuples as value.
    """

    # ---- Integration of Gemini's code for cycling through subfolders ----
    print(f"SIFT detection (kp+des) from: {dataset_path}")
    db = {}
    entries = os.listdir(dataset_path)
    subfolders = [f for f in entries if os.path.isdir(os.path.join(dataset_path, f))]
    for card_id in subfolders:
        card_path = os.path.join(dataset_path, card_id)
        db[card_id] = []
        original_images_for_type = []
        dummy_labels = []
        for img_name in os.listdir(card_path):
            img_file_path = os.path.join(card_path, img_name)
            img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                original_images_for_type.append(img_resized)
                dummy_labels.append(0)
            if not original_images_for_type:
                print(f"No images for the card '{card_id}'")
                continue
            images_to_process = np.array(original_images_for_type)
            if len(original_images_for_type) > 0:
                print(f"  Augmenting '{card_id}' (M={augmentation_m})...")
                images_to_process, _ = augment_data(images_to_process, np.array(dummy_labels), augmentation_m)
            for proc_img in images_to_process:
                if proc_img.dtype != np.uint8:
                    proc_img = proc_img.astype(np.uint8)
                kp, des = extract_sift_features(proc_img)
                if des is not None and len(kp) > 0:
                    db[card_id].append((kp, des))
            if db[card_id]:
                print(f"  Loaded {len(db[card_id])} set of (kp, des) for '{card_id}'.")
            else:
                print(f"  ATTENTION: no feature SIFT for '{card_id}'.")
    if not db: print("ERROR: No SIFT features found in any subfolder.")
    return db


def identify_single_card(card_roi_bgr, faceup_sift_db_kp_des, config):
    """
    Identifies a single card ROI using SIFT and RANSAC.
    Returns the card state ("face-up" or "face-down"), card ID, and number of inliers.
    """

    if card_roi_bgr is None or card_roi_bgr.size == 0 or not faceup_sift_db_kp_des:
        return "face-down", None, 0

    roi_gray = cv2.cvtColor(card_roi_bgr, cv2.COLOR_BGR2GRAY)
    roi_gray_resized = cv2.resize(roi_gray, config['REFERENCE_IMAGE_SIZE']) 

    kp_roi, des_roi = extract_sift_features(roi_gray_resized) 

    if des_roi is None or len(kp_roi) < 3:
        return "face-down", None, 0 

    best_match_card_id = None
    max_inliers_overall = 0

    for card_id_ref, kp_des_pairs_for_ref_type in faceup_sift_db_kp_des.items():
        current_ref_type_max_inliers = 0
        for kp_ref, des_ref in kp_des_pairs_for_ref_type: 
            if des_ref is None or len(kp_ref) < 3:
                continue 
            try:
                good_matches_list, _ = match_descriptors(des_ref, des_roi)
            except Exception as e:
                continue

            if len(good_matches_list) < config['MIN_RAW_MATCHES_BEFORE_RANSAC']:
                continue

            pts_ref_match = np.float32([kp_ref[match_obj[0].queryIdx].pt for match_obj in good_matches_list]).T 
            pts_roi_match = np.float32([kp_roi[match_obj[0].trainIdx].pt for match_obj in good_matches_list]).T

            if pts_ref_match.shape[1] < 3:
                continue

            A_est, t_est = ransac_fit_affine(pts_ref_match, pts_roi_match, 
                                             threshold=config['RANSAC_THRESHOLD_PIXELS'], 
                                             n_iter=config['RANSAC_ITERATIONS'])

            num_inliers_for_this_ref_img = 0

            if A_est is not None and t_est is not None:
                res_lengths = residual_lgths(A_est, t_est.reshape(2,1), pts_ref_match, pts_roi_match)
                num_inliers_for_this_ref_img = np.sum(res_lengths < config['RANSAC_THRESHOLD_PIXELS'])

            if num_inliers_for_this_ref_img > current_ref_type_max_inliers:
                current_ref_type_max_inliers = num_inliers_for_this_ref_img

        if current_ref_type_max_inliers > max_inliers_overall:
            max_inliers_overall = current_ref_type_max_inliers
            best_match_card_id = card_id_ref
    
 
    if max_inliers_overall >= config['MIN_INLIER_COUNT_FOR_ID']:
        #print(f"Card identified as '{best_match_card_id}' with {max_inliers_overall} inliers.")
        return "face-up", best_match_card_id, max_inliers_overall
    else:
        return "face-down", None, max_inliers_overall

def load_and_detect_cards(image_path, expected_cards):
    """
    Loads an image and detects cards using blob detection.
    Returns the image for drawing, detected card bounding boxes, and the original image.
    """

    if not os.path.exists(image_path):
        print(f"ERROR: The image in '{image_path}' has not be found.")
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_for_drawing_display, detected_card_bboxes_coords, original_game_bgr = \
        find_cards_in_image_blob(
            image_path,
            expected_cards
        )

    if image_for_drawing_display is None or not detected_card_bboxes_coords:
        print(f"Impossible to detect photos from '{image_path}'.")
        raise RuntimeError(f"Blob detection failed for '{image_path}'.")
    
    return image_for_drawing_display, detected_card_bboxes_coords, original_game_bgr

def analyze_board(detected_card_bboxes_coords, original_game_bgr, faceup_sift_db, sift_config):
    """
    Analyzes the board and identifies cards using SIFT and RANSAC.
    Returns a list of dictionaries with card status, ID, and bounding box.
    """
    board_cards_status = []
    for i, contour in enumerate(detected_card_bboxes_coords):
        x, y, w, h = cv2.boundingRect(contour)
        card_roi_from_board = original_game_bgr[y:y+h, x:x+w]
        if card_roi_from_board.size == 0:
            status = {'index': i, 'state': 'face-down', 'id': None, 'bbox': (x,y,w,h), 'inliers': 0}
            board_cards_status.append(status)
            continue

        state, card_id, inliers = identify_single_card(
            card_roi_from_board,
            faceup_sift_db,
            sift_config
        )
        board_cards_status.append({'index': i, 'state': state, 'id': card_id, 'bbox': (x,y,w,h), 'inliers': inliers})
    return board_cards_status

def find_pairs(face_up_cards_on_board):
    """
    Finds pairs of cards on the board that are face-up and have the same ID.
    Returns a list of tuples with the found pairs.
    """
    found_pairs_info = []
    if len(face_up_cards_on_board) == 2:
        c1 = face_up_cards_on_board[0]
        c2 = face_up_cards_on_board[1]
        if c1['id'] == c2['id']:
            found_pairs_info.append((c1, c2))
            #print(f"\nFOUND COUPLE  {c1['id']} (idx {c1['index']+1}) e {c2['id']} (idx {c2['index']+1})")
        else:
            #print(f"\nNO COUPLE  {c1['id']} (idx {c1['index']+1}) e {c2['id']} (idx {c2['index']+1})")
            pass
    return found_pairs_info

# ------ Function from Gemini   ------
def mapping_images_in_subfolder_with_number(path_subfolder: str) -> List[Tuple[int, str]]:
    """
    Mapping of images in a subfolder with a number.
    This function scans the specified subfolder for image files and returns a list of tuples
    containing the index and the name of each image file found.
    The images are sorted by name to ensure a consistent numerical assignment.
    """

    mapped_images: List[Tuple[int, str]] = []
    image_extension = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp')

    if not os.path.isdir(path_subfolder):
        print(f"Attention: The specified path '{path_subfolder}' is not a valid directory or does not exist.")
        return mapped_images

    image_counter = 0
    #print(f"Search images in: {path_subfolder}")

    # Scans the subfolder for image files
    # and maps them with a number
    for nome_file in sorted(os.listdir(path_subfolder)):
        file_path = os.path.join(path_subfolder, nome_file)
        
        # Check if the file is an image
        if os.path.isfile(file_path) and nome_file.lower().endswith(image_extension):
            image_counter += 1
            mapped_images.append((image_counter, nome_file))
            #print(f"  Found image {contatore_immagini}: {nome_file}")

    if not mapped_images:
        print(f"Attention: No images found in the specified path '{path_subfolder}'.")
    return mapped_images
#--- End of Gemini's code ------


def draw_and_show_board(image_to_draw_on, board_cards_status, found_pairs_info, 
                        rect_color_face_down, rect_color_face_up_match, rect_color_face_up_nomatch, 
                        rectangle_thickness, title="Board_anaysis"):
   
    display_img_bgr = image_to_draw_on.copy()
    pair_card_indices_this_turn = []
    if found_pairs_info: 
        pair_card_indices_this_turn = [found_pairs_info[0][0]['index'], found_pairs_info[0][1]['index']]

    for card_info in board_cards_status: 
        x, y, w, h = card_info['bbox']
        label_text = f"{card_info['index']+1}"
        rect_color = rect_color_face_down 

        if card_info['state'] == "face-up":
            card_id_display = card_info['id'] if card_info['id'] else 'Unk'
            label_text += f":{card_id_display}(I:{card_info['inliers']})"

            if card_info['index'] in pair_card_indices_this_turn:
                rect_color = rect_color_face_up_match
            else:
                rect_color = rect_color_face_up_nomatch
        else: 
            label_text += f":FD(I:{card_info['inliers']})"

        
        cv2.rectangle(display_img_bgr, (x, y), (x + w, y + h), rect_color, rectangle_thickness)

        label_size_txt, base_line_txt = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

      
        cv2.rectangle(display_img_bgr, 
                      (x, y - label_size_txt[1] - base_line_txt - 2), 
                      (x + label_size_txt[0] + 4, y - base_line_txt + 2), 
                      (0,0,0), -1) 
     
        cv2.putText(display_img_bgr, label_text, 
                    (x + 2, y - base_line_txt -1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA) 


    display_img_rgb = cv2.cvtColor(display_img_bgr, cv2.COLOR_BGR2RGB)

    
    plt.figure(figsize=(5, 5)) 
    plt.imshow(display_img_rgb)
    plt.title(title)
    plt.axis('off') 
    plt.show() 
    return

def analyze_board_for_verification(detected_card_bboxes_list, original_game_bgr, faceup_sift_db, sift_config):
    """
    Analyses each detected card on the board (given as a list of bounding boxes).
    Returns a list of dictionaries with the state and ID of each card.
    Same used before but with a different input format. 
    """
    board_cards_status = []
    for i, bbox in enumerate(detected_card_bboxes_list): 
        x, y, w, h = bbox  

    
        x, y, w, h = int(x), int(y), int(w), int(h)

        if w <= 0 or h <= 0: 
            status = {'index': i, 'state': 'error_roi', 'id': None, 'bbox': (x,y,w,h), 'inliers': 0}
            board_cards_status.append(status)
            continue
            
        card_roi_from_board = original_game_bgr[y:y+h, x:x+w]
        
        if card_roi_from_board.size == 0:
            status = {'index': i, 'state': 'face-down', 'id': None, 'bbox': (x,y,w,h), 'inliers': 0}
            board_cards_status.append(status)
            continue

        state, card_id, inliers = identify_single_card(
            card_roi_from_board,
            faceup_sift_db,
            sift_config 
        )
        board_cards_status.append({'index': i, 'state': state, 'id': card_id, 'bbox': (x,y,w,h), 'inliers': inliers})
    return board_cards_status