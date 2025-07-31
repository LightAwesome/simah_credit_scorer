import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import camelot
from collections import defaultdict
import io

def convert_pdf_to_image(pdf_path, page_num, dpi=300):
    """Convert PDF page to high-quality image for OCR and contour detection"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed

    # High DPI for better OCR accuracy and contour detection
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")

    # Convert to PIL Image
    img = Image.open(io.BytesIO(img_data))
    doc.close()

    return img, dpi

def detect_high_contrast_boundaries(gray_img, dpi=300):
    """
    Detect high contrast boundaries that typically indicate table borders
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Enhanced edge detection with multiple techniques

    # 1. Canny edge detection with optimized parameters
    edges_canny = cv2.Canny(blurred, 30, 80, apertureSize=3, L2gradient=True)

    # 2. Sobel edge detection for horizontal and vertical edges
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined * 255 / np.max(sobel_combined))

    # 3. Laplacian edge detection for fine details
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # 4. Morphological gradient for structural boundaries
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_morph)

    # Combine all edge detection methods
    combined_edges = cv2.bitwise_or(edges_canny,
                     cv2.bitwise_or(cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)[1],
                     cv2.bitwise_or(cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)[1],
                                   cv2.threshold(morph_gradient, 40, 255, cv2.THRESH_BINARY)[1])))

    return combined_edges, edges_canny, sobel_combined, laplacian, morph_gradient

def enhance_rectangular_structures(combined_edges, dpi=300):
    """
    Enhance rectangular structures that are likely to be tables
    """
    # Create kernels for detecting horizontal and vertical lines
    scale_factor = max(1, int(dpi / 150))  # Scale based on DPI

    # Horizontal line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40 * scale_factor, 1))
    horizontal_lines = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, horizontal_kernel)

    # Vertical line detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40 * scale_factor))
    vertical_lines = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, vertical_kernel)

    # Combine horizontal and vertical lines
    table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Enhance intersections (corners of tables)
    intersection_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    intersections = cv2.morphologyEx(table_structure, cv2.MORPH_CLOSE, intersection_kernel)

    # Combine original edges with enhanced structures
    enhanced = cv2.bitwise_or(combined_edges, cv2.bitwise_or(table_structure, intersections))

    # Clean up noise while preserving table structures
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, cleanup_kernel)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, cleanup_kernel)

    return enhanced, horizontal_lines, vertical_lines, intersections

def detect_table_boundaries_with_contours(img, dpi=300):
    """
    Enhanced table boundary detection using high contrast contour checking
    """
    # Convert PIL to OpenCV format
    opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

    # Apply high contrast boundary detection
    combined_edges, edges_canny, sobel_combined, laplacian, morph_gradient = detect_high_contrast_boundaries(gray, dpi)

    # Enhance rectangular structures
    enhanced_edges, horizontal_lines, vertical_lines, intersections = enhance_rectangular_structures(combined_edges, dpi)

    # Apply additional threshold to get binary image for dark text/lines
    _, thresh_dark = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Combine enhanced edges with threshold
    final_combined = cv2.bitwise_or(enhanced_edges, thresh_dark)

    # Morphological operations to connect nearby components
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_combined = cv2.morphologyEx(final_combined, cv2.MORPH_CLOSE, connect_kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(final_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find table-like rectangles with enhanced criteria
    table_contours = []

    # Dynamic thresholds based on image size
    img_area = gray.shape[0] * gray.shape[1]
    min_area = max(5000, img_area * 0.01)  # At least 1% of image or 5000 pixels
    min_aspect_ratio = 0.15  # Allow more narrow tables
    max_aspect_ratio = 8.0   # Allow wider tables

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Check if it looks like a table (rectangular with reasonable aspect ratio)
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # Enhanced rectangular check
                epsilon = 0.015 * cv2.arcLength(contour, True)  # More lenient approximation
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Calculate how "rectangular" the contour is
                rect_area = w * h
                fill_ratio = area / rect_area if rect_area > 0 else 0

                # More lenient criteria for table detection
                if len(approx) >= 4 and fill_ratio > 0.3:  # At least somewhat rectangular
                    # Check for high contrast boundaries within the region
                    roi_edges = enhanced_edges[y:y+h, x:x+w]
                    edge_density = np.sum(roi_edges > 0) / (w * h) if (w * h) > 0 else 0

                    # Calculate contrast score
                    roi_gray = gray[y:y+h, x:x+w]
                    contrast_score = calculate_contrast_score(roi_gray)

                    # Enhanced scoring system
                    boundary_score = calculate_boundary_score(
                        area, aspect_ratio, fill_ratio, edge_density, contrast_score,
                        roi_edges, horizontal_lines[y:y+h, x:x+w], vertical_lines[y:y+h, x:x+w]
                    )

                    if boundary_score > 0.3:  # Threshold for accepting as potential table
                        table_contours.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'fill_ratio': fill_ratio,
                            'edge_density': edge_density,
                            'contrast_score': contrast_score,
                            'boundary_score': boundary_score
                        })

    # Sort by boundary score (highest first) to prioritize best candidates
    table_contours.sort(key=lambda x: x['boundary_score'], reverse=True)

    # Return debug images for visualization
    debug_images = {
        'combined_edges': combined_edges,
        'enhanced_edges': enhanced_edges,
        'horizontal_lines': horizontal_lines,
        'vertical_lines': vertical_lines,
        'intersections': intersections,
        'final_combined': final_combined
    }

    return table_contours, debug_images

def calculate_contrast_score(roi_gray):
    """
    Calculate contrast score for a region of interest
    """
    if roi_gray.size == 0:
        return 0

    # Calculate standard deviation (measure of contrast)
    std_dev = np.std(roi_gray)

    # Calculate difference between max and min values
    value_range = np.max(roi_gray) - np.min(roi_gray)

    # Normalize scores
    contrast_score = min(1.0, (std_dev / 50.0 + value_range / 255.0) / 2.0)

    return contrast_score

def calculate_boundary_score(area, aspect_ratio, fill_ratio, edge_density, contrast_score,
                           roi_edges, roi_horizontal, roi_vertical):
    """
    Calculate comprehensive boundary score for table detection
    """
    # Area score (larger areas are more likely to be tables)
    area_score = min(1.0, area / 50000)

    # Aspect ratio score (prefer reasonable table proportions)
    if 0.3 <= aspect_ratio <= 4.0:
        aspect_score = 1.0
    elif 0.15 <= aspect_ratio <= 8.0:
        aspect_score = 0.7
    else:
        aspect_score = 0.3

    # Fill ratio score (prefer more rectangular shapes)
    fill_score = min(1.0, fill_ratio * 2)  # Boost fill ratio importance

    # Edge density score (tables should have visible boundaries)
    edge_score = min(1.0, edge_density * 10)

    # Structure score (check for horizontal and vertical lines)
    h_lines = np.sum(roi_horizontal > 0)
    v_lines = np.sum(roi_vertical > 0)
    total_pixels = roi_horizontal.size
    structure_score = min(1.0, (h_lines + v_lines) / (total_pixels * 0.1)) if total_pixels > 0 else 0

    # Weighted combination
    boundary_score = (
        area_score * 0.2 +        # Size importance
        aspect_score * 0.15 +     # Shape importance
        fill_score * 0.2 +        # Rectangularity
        edge_score * 0.2 +        # Edge visibility
        contrast_score * 0.15 +   # Contrast
        structure_score * 0.1     # Line structure
    )

    return min(1.0, boundary_score)

def detect_text_within_boundaries(img, table_contours, dpi=300):
    """
    Enhanced OCR validation with better text detection
    """
    validated_tables = []

    for i, table_info in enumerate(table_contours):
        x, y, w, h = table_info['bbox']

        # Extract the region of interest with padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.width, x + w + padding)
        y_end = min(img.height, y + h + padding)

        roi = img.crop((x_start, y_start, x_end, y_end))

        # Enhance ROI for better OCR
        roi_enhanced = enhance_roi_for_ocr(roi)

        # Get OCR data for this region
        try:
            # Use multiple OCR configurations
            ocr_configs = [
                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()%-+/:$ ',
                '--psm 4',
                '--psm 3'
            ]

            best_text_elements = []
            best_count = 0

            for config in ocr_configs:
                try:
                    ocr_data = pytesseract.image_to_data(roi_enhanced, output_type=pytesseract.Output.DICT, config=config)

                    text_elements = []
                    for j in range(len(ocr_data['text'])):
                        confidence = int(ocr_data['conf'][j])
                        text = ocr_data['text'][j].strip()

                        if confidence > 25 and text and len(text) > 1:  # Lowered confidence threshold
                            text_elements.append({
                                'text': text,
                                'x': ocr_data['left'][j],
                                'y': ocr_data['top'][j],
                                'w': ocr_data['width'][j],
                                'h': ocr_data['height'][j],
                                'confidence': confidence
                            })

                    if len(text_elements) > best_count:
                        best_text_elements = text_elements
                        best_count = len(text_elements)

                except Exception as ocr_error:
                    continue

            # Validate if this boundary contains table-like content
            if len(best_text_elements) >= 3:  # At least 3 text elements
                # Enhanced row/column detection
                rows, columns = analyze_text_structure(best_text_elements)

                # Calculate enhanced confidence
                enhanced_confidence = calculate_enhanced_boundary_confidence(
                    table_info, best_text_elements, rows, columns
                )

                if enhanced_confidence > 0.25:  # Lower threshold for more detection
                    validated_tables.append({
                        'bbox': table_info['bbox'],
                        'area': table_info['area'],
                        'text_elements': len(best_text_elements),
                        'rows': len(rows),
                        'columns': len(columns),
                        'boundary_score': table_info['boundary_score'],
                        'confidence': enhanced_confidence,
                        'contrast_score': table_info['contrast_score'],
                        'edge_density': table_info['edge_density']
                    })

        except Exception as e:
            print(f"OCR error for boundary {i}: {e}")
            continue

    return validated_tables

def enhance_roi_for_ocr(roi):
    """
    Enhance ROI image for better OCR performance
    """
    # Convert to numpy array
    roi_np = np.array(roi)

    # Convert to grayscale if needed
    if len(roi_np.shape) == 3:
        roi_gray = cv2.cvtColor(roi_np, cv2.COLOR_RGB2GRAY)
    else:
        roi_gray = roi_np

    # Apply different enhancement techniques

    # 1. Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi_gray)

    # 2. Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 3. Sharpening
    kernel_sharpen = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # 4. Ensure proper contrast
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)

    # Convert back to PIL Image
    return Image.fromarray(enhanced)

def analyze_text_structure(text_elements):
    """
    Enhanced analysis of text structure to identify rows and columns
    """
    if not text_elements:
        return [], []

    # Sort by y-coordinate to group into rows
    text_elements.sort(key=lambda t: t['y'])

    # Group into rows with adaptive tolerance
    rows = []
    current_row = []
    last_y = None

    # Calculate adaptive row tolerance based on text heights
    heights = [elem['h'] for elem in text_elements if elem['h'] > 0]
    avg_height = sum(heights) / len(heights) if heights else 20
    row_tolerance = max(10, int(avg_height * 0.5))

    for elem in text_elements:
        if last_y is None or abs(elem['y'] - last_y) <= row_tolerance:
            current_row.append(elem)
        else:
            if current_row:
                # Sort row by x-coordinate
                current_row.sort(key=lambda e: e['x'])
                rows.append(current_row)
            current_row = [elem]
        last_y = elem['y']

    if current_row:
        current_row.sort(key=lambda e: e['x'])
        rows.append(current_row)

    # Analyze column structure
    columns = []
    if rows:
        # Find potential column positions based on x-coordinates
        all_x_positions = []
        for row in rows:
            for elem in row:
                all_x_positions.append(elem['x'])

        # Cluster x-positions to find column boundaries
        all_x_positions.sort()
        column_positions = []
        last_pos = None
        tolerance = 30  # Pixel tolerance for column alignment

        for pos in all_x_positions:
            if last_pos is None or pos - last_pos > tolerance:
                column_positions.append(pos)
                last_pos = pos

        columns = column_positions

    return rows, columns

def calculate_enhanced_boundary_confidence(table_info, text_elements, rows, columns):
    """
    Calculate enhanced confidence score with multiple factors
    """
    # Base boundary score from contour analysis
    boundary_score = table_info['boundary_score']

    # Text density score
    area = table_info['area']
    text_density = len(text_elements) / (area / 1000) if area > 0 else 0
    text_score = min(1.0, text_density / 3)

    # Structure scores
    row_score = min(1.0, len(rows) / 8) if rows else 0
    column_score = min(1.0, len(columns) / 6) if columns else 0

    # Multi-column row consistency
    multi_col_rows = sum(1 for row in rows if len(row) >= 2) if rows else 0
    consistency_score = multi_col_rows / len(rows) if rows else 0

    # Text quality score based on OCR confidence
    if text_elements:
        avg_confidence = sum(elem['confidence'] for elem in text_elements) / len(text_elements)
        quality_score = min(1.0, avg_confidence / 60)  # Normalize from 0-100 scale
    else:
        quality_score = 0

    # Enhanced weighted combination
    enhanced_confidence = (
        boundary_score * 0.3 +      # Visual boundary detection
        text_score * 0.2 +          # Text density
        row_score * 0.15 +          # Row detection
        column_score * 0.15 +       # Column detection
        consistency_score * 0.1 +   # Structure consistency
        quality_score * 0.1         # OCR quality
    )

    return min(1.0, enhanced_confidence)

def save_debug_image(img, table_contours, validated_tables, page_num, debug_images=None):
    """
    Enhanced debug image saving with multiple visualization layers
    """
    debug_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Draw all detected contours in red (potential tables)
    for table_info in table_contours:
        x, y, w, h = table_info['bbox']
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add detailed information
        info_text = f"Score: {table_info['boundary_score']:.2f}"
        cv2.putText(debug_img, info_text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Add contrast and edge info
        detail_text = f"C:{table_info['contrast_score']:.1f} E:{table_info['edge_density']:.2f}"
        cv2.putText(debug_img, detail_text, (x, y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw validated tables in green (final selections)
    for i, table in enumerate(validated_tables):
        x, y, w, h = table['bbox']
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Add validation information
        validation_text = f"Table {i+1}: {table['confidence']:.2f}"
        cv2.putText(debug_img, validation_text, (x, y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add structure info
        structure_text = f"R:{table['rows']} C:{table['columns']} T:{table['text_elements']}"
        cv2.putText(debug_img, structure_text, (x, y - 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save main debug image
    debug_filename = f"debug_page_{page_num}_enhanced_boundaries.jpg"
    cv2.imwrite(debug_filename, debug_img)
    print(f"Enhanced debug image saved: {debug_filename}")

    # Save additional debug images if available
    if debug_images:
        for name, img_data in debug_images.items():
            if img_data is not None:
                filename = f"debug_page_{page_num}_{name}.jpg"
                cv2.imwrite(filename, img_data)
                print(f"Debug layer saved: {filename}")

def convert_to_pdf_coordinates(bbox, img_height, dpi=300):
    """Convert image coordinates to PDF coordinates"""
    scale_factor = 72 / dpi  # PDF uses 72 DPI

    x, y, w, h = bbox

    # Convert and flip Y coordinate (PDF origin is bottom-left)
    pdf_x1 = x * scale_factor
    pdf_y1 = (img_height - y - h) * scale_factor  # Flip Y
    pdf_x2 = (x + w) * scale_factor
    pdf_y2 = (img_height - y) * scale_factor  # Flip Y

    return (pdf_x1, pdf_y1, pdf_x2, pdf_y2)

def detect_tables_with_boundaries_and_ocr(pdf_path, page_num, dpi=300):
    """Enhanced main function with high contrast boundary detection"""
    print(f"Processing page {page_num} with enhanced boundary + OCR detection...")

    # Convert PDF to image
    img, dpi = convert_pdf_to_image(pdf_path, page_num, dpi)

    # Detect table boundaries using enhanced contour detection
    table_contours, debug_images = detect_table_boundaries_with_contours(img, dpi)
    print(f"Found {len(table_contours)} potential table boundaries")

    # Print detailed boundary information
    for i, contour_info in enumerate(table_contours):
        print(f"  Boundary {i+1}: Score={contour_info['boundary_score']:.3f}, "
              f"Area={contour_info['area']:.0f}, Contrast={contour_info['contrast_score']:.2f}, "
              f"Edges={contour_info['edge_density']:.3f}")

    # Validate boundaries with enhanced OCR
    validated_tables = detect_text_within_boundaries(img, table_contours, dpi)
    print(f"Validated {len(validated_tables)} tables with enhanced OCR")

    # Convert to PDF coordinates and format for Camelot
    table_areas = []
    for i, table in enumerate(validated_tables):
        bbox = table['bbox']
        confidence = table['confidence']

        # Convert to PDF coordinates
        pdf_coords = convert_to_pdf_coordinates(bbox, img.height, dpi)

        # Format as Camelot table_areas string: "x1,y1,x2,y2"
        area_string = f"{pdf_coords[0]:.0f},{pdf_coords[1]:.0f},{pdf_coords[2]:.0f},{pdf_coords[3]:.0f}"

        table_areas.append({
            'area': area_string,
            'confidence': confidence,
            'bbox': bbox,  # Keep original for debugging
            'text_elements': table['text_elements'],
            'rows': table['rows'],
            'columns': table['columns'],
            'boundary_score': table['boundary_score'],
            'contrast_score': table['contrast_score'],
            'edge_density': table['edge_density']
        })

        print(f"\nTable {i+1}: {area_string}")
        print(f"  - Overall Confidence: {confidence:.3f}")
        print(f"  - Boundary Score: {table['boundary_score']:.3f}")
        print(f"  - Contrast Score: {table['contrast_score']:.3f}")
        print(f"  - Edge Density: {table['edge_density']:.3f}")
        print(f"  - Text elements: {table['text_elements']}")
        print(f"  - Rows: {table['rows']}, Columns: {table['columns']}")

    # Save enhanced debug images
    #save_debug_image(img, table_contours, validated_tables, page_num, debug_images)

    return table_areas

def process_all_pages_boundary_ocr(pdf_path, dpi=300, save_debug=True):
    """Process all pages in PDF with enhanced boundary detection"""
    doc = fitz.open(pdf_path)
    page_count = doc.page_count
    doc.close()

    all_table_areas = {}

    for page_num in range(1, page_count + 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING PAGE {page_num} - ENHANCED HIGH CONTRAST DETECTION")
        print(f"{'='*60}")

        try:
            table_areas = detect_tables_with_boundaries_and_ocr(pdf_path, page_num, dpi)
            all_table_areas[page_num] = table_areas

            print(f"\nSUMMARY - Page {page_num}: Found {len(table_areas)} validated tables")
            for i, table in enumerate(table_areas):
                print(f"  Table {i+1}: {table['area']}")
                print(f"    Confidence: {table['confidence']:.3f} | "
                      f"Boundary: {table['boundary_score']:.3f} | "
                      f"Contrast: {table['contrast_score']:.3f}")

        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
            all_table_areas[page_num] = []

    return all_table_areas

def verify_detected_tables(pdf_path, all_table_areas, print_full_tables=True):
    """Verify detected tables by actually reading them with Camelot and printing full DataFrames"""
    print(f"\n{'='*60}")
    print("VERIFYING DETECTED TABLES")
    print(f"{'='*60}")

    all_extracted_tables = {}

    for page_num, areas in all_table_areas.items():
        print(f"\nPAGE {page_num} TABLES:")
        print("-" * 40)

        page_tables = []

        for i, table_info in enumerate(areas):
            area = table_info['area']
            confidence = table_info['confidence']

            print(f"\nTable {i+1} - Coordinates: {area}")
            print(f"Confidence: {confidence:.3f} | Boundary: {table_info['boundary_score']:.3f}")
            print("-" * 30)

            try:
                # Try stream flavor first
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num),
                    flavor="stream",
                    table_areas=[area]
                )

                if tables and len(tables) > 0:
                    df = tables[0].df

                    # Clean up the DataFrame
                    # Remove completely empty rows and columns
                    df = df.dropna(how='all').dropna(axis=1, how='all')

                    # Reset index
                    df = df.reset_index(drop=True)

                    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    print("Extraction: SUCCESS")

                    if print_full_tables:
                        print("\nFULL TABLE CONTENT:")
                        print(df.to_string(index=False, max_rows=None, max_cols=None))
                    else:
                        print("\nPREVIEW (first 5 rows):")
                        print(df.head().to_string(index=False))

                    # Check if table looks properly parsed
                    non_empty_cells = df.notna().sum().sum()
                    total_cells = df.size
                    fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0

                    print(f"\nTable Quality Metrics:")
                    print(f"  - Fill rate: {fill_rate:.1%} ({non_empty_cells}/{total_cells} cells with data)")
                    print(f"  - Column consistency: {check_column_consistency(df)}")

                    page_tables.append({
                        'table_num': i+1,
                        'coordinates': area,
                        'dataframe': df,
                        'confidence': confidence,
                        'boundary_score': table_info['boundary_score'],
                        'contrast_score': table_info['contrast_score'],
                        'fill_rate': fill_rate,
                        'extraction_status': 'SUCCESS'
                    })

                else:
                    print("Shape: 0 rows × 0 columns")
                    print("Extraction: NO DATA FOUND")

                    # Try lattice flavor as backup
                    print("Trying 'lattice' flavor...")
                    try:
                        tables_lattice = camelot.read_pdf(
                            pdf_path,
                            pages=str(page_num),
                            flavor="lattice",
                            table_areas=[area]
                        )

                        if tables_lattice and len(tables_lattice) > 0:
                            df = tables_lattice[0].df
                            df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)

                            print(f"Lattice extraction - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                            if print_full_tables:
                                print("\nFULL TABLE CONTENT (LATTICE):")
                                print(df.to_string(index=False, max_rows=None, max_cols=None))

                            page_tables.append({
                                'table_num': i+1,
                                'coordinates': area,
                                'dataframe': df,
                                'confidence': confidence,
                                'boundary_score': table_info['boundary_score'],
                                'contrast_score': table_info['contrast_score'],
                                'fill_rate': df.notna().sum().sum() / df.size if df.size > 0 else 0,
                                'extraction_status': 'SUCCESS (lattice)'
                            })
                        else:
                            print("Lattice extraction also failed")
                            page_tables.append({
                                'table_num': i+1,
                                'coordinates': area,
                                'dataframe': None,
                                'confidence': confidence,
                                'boundary_score': table_info['boundary_score'],
                                'contrast_score': table_info['contrast_score'],
                                'fill_rate': 0,
                                'extraction_status': 'FAILED'
                            })
                    except Exception as lattice_error:
                        print(f"Lattice extraction error: {lattice_error}")
                        page_tables.append({
                            'table_num': i+1,
                            'coordinates': area,
                            'dataframe': None,
                            'confidence': confidence,
                            'boundary_score': table_info['boundary_score'],
                            'contrast_score': table_info['contrast_score'],
                            'fill_rate': 0,
                            'extraction_status': 'FAILED'
                        })

            except Exception as e:
                print(f"Extraction: ERROR - {e}")
                page_tables.append({
                    'table_num': i+1,
                    'coordinates': area,
                    'dataframe': None,
                    'confidence': confidence,
                    'boundary_score': table_info.get('boundary_score', 0),
                    'contrast_score': table_info.get('contrast_score', 0),
                    'fill_rate': 0,
                    'extraction_status': f'ERROR: {str(e)}'
                })

        all_extracted_tables[page_num] = page_tables

    return all_extracted_tables

def check_column_consistency(df):
    """Check if columns have consistent data types (indicator of good parsing)"""
    if df.empty:
        return "N/A (empty table)"

    consistent_cols = 0
    total_cols = len(df.columns)

    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) > 1:
            # Check if column contains mostly numbers, text, or mixed
            numeric_count = sum(1 for x in col_data if str(x).replace('.', '').replace('-', '').replace(',', '').isdigit())
            if numeric_count > len(col_data) * 0.8 or numeric_count < len(col_data) * 0.2:
                consistent_cols += 1

    return f"{consistent_cols}/{total_cols} columns consistent"

def print_extraction_summary(all_extracted_tables):
    """Print a comprehensive summary of all extractions with enhanced metrics"""
    print(f"\n{'='*60}")
    print("ENHANCED EXTRACTION SUMMARY")
    print(f"{'='*60}")

    total_tables = 0
    successful_extractions = 0
    total_boundary_score = 0
    total_contrast_score = 0

    for page_num, page_tables in all_extracted_tables.items():
        print(f"\nPage {page_num}:")
        for table in page_tables:
            total_tables += 1
            status = "✓" if table['extraction_status'].startswith('SUCCESS') else "✗"
            fill_rate = f"{table['fill_rate']:.1%}" if table['fill_rate'] > 0 else "0%"
            boundary_score = table.get('boundary_score', 0)
            contrast_score = table.get('contrast_score', 0)

            total_boundary_score += boundary_score
            total_contrast_score += contrast_score

            print(f"  {status} Table {table['table_num']}: {table['extraction_status']}")
            print(f"    Fill rate: {fill_rate} | Boundary: {boundary_score:.3f} | Contrast: {contrast_score:.3f}")

            if table['extraction_status'].startswith('SUCCESS'):
                successful_extractions += 1

    success_rate = successful_extractions / total_tables * 100 if total_tables > 0 else 0
    avg_boundary_score = total_boundary_score / total_tables if total_tables > 0 else 0
    avg_contrast_score = total_contrast_score / total_tables if total_tables > 0 else 0

    print(f"\nOverall Metrics:")
    print(f"  Success Rate: {successful_extractions}/{total_tables} ({success_rate:.1f}%)")
    print(f"  Average Boundary Score: {avg_boundary_score:.3f}")
    print(f"  Average Contrast Score: {avg_contrast_score:.3f}")

def save_tables_to_csv(all_extracted_tables, output_dir="extracted_tables_enhanced"):
    """Save all successfully extracted tables to CSV files with enhanced naming"""
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_files = []

    for page_num, page_tables in all_extracted_tables.items():
        for table in page_tables:
            if table['dataframe'] is not None and not table['dataframe'].empty:
                # Enhanced filename with quality metrics
                boundary_score = table.get('boundary_score', 0)
                contrast_score = table.get('contrast_score', 0)

                filename = f"page_{page_num}_table_{table['table_num']}_b{boundary_score:.2f}_c{contrast_score:.2f}.csv"
                filepath = os.path.join(output_dir, filename)
                table['dataframe'].to_csv(filepath, index=False)
                saved_files.append(filepath)
                print(f"Saved: {filepath}")

    return saved_files
def save_tables_to_text(all_extracted_tables, output_file="extracted_tables_enhanced/all_tables.txt"):
    """Save all successfully extracted tables into a single text file with structured metadata"""

    import os

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lines = []

    for page_num, page_tables in all_extracted_tables.items():
        for table in page_tables:
            df = table.get("dataframe")
            if df is not None and not df.empty:
                boundary_score = table.get("boundary_score", 0)
                contrast_score = table.get("contrast_score", 0)
                table_num = table.get("table_num", 0)

                # Metadata block
                lines.append("=" * 80)
                lines.append(f"Page {page_num} - Table {table_num}")
                lines.append(f"Boundary Score: {boundary_score:.2f}")
                lines.append(f"Contrast Score: {contrast_score:.2f}")
                lines.append("-" * 80)

                # Table as text
                table_text = df.to_string(index=False)
                lines.append(table_text)
                lines.append("\n")

    # Write to text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved all tables to text file: {output_file}")
    return output_file

def generate_camelot_code_from_boundaries(pdf_path, all_table_areas, table_names=None):
    """Generate Camelot code with enhanced boundary-detected coordinates"""
    code_lines = []
    name_index = 0

    # Default table names if not provided
    if table_names is None:
        table_names = []
        for page_num, areas in all_table_areas.items():
            for i in range(len(areas)):
                table_names.append(f"table_p{page_num}_{i+1}")

    for page_num, areas in all_table_areas.items():
        for i, table_info in enumerate(areas):
            if name_index < len(table_names):
                var_name = table_names[name_index]
            else:
                var_name = f"table_p{page_num}_{i+1}"

            area = table_info['area']
            confidence = table_info['confidence']
            boundary_score = table_info['boundary_score']
            contrast_score = table_info['contrast_score']
            text_elements = table_info['text_elements']

            code = f'''{var_name} = camelot.read_pdf(
    "{pdf_path}",
    pages="{page_num}",
    flavor="stream",  # try "lattice" if borders are visible
    table_areas=["{area}"]
    # Enhanced Detection Metrics:
    # - Confidence: {confidence:.3f}
    # - Boundary Score: {boundary_score:.3f}
    # - Contrast Score: {contrast_score:.3f}
    # - Text Elements: {text_elements}
)'''
            code_lines.append(code)
            name_index += 1

    return "\n\n".join(code_lines)

# Main execution
def extract(pdf_path):

    print("ENHANCED HIGH CONTRAST BOUNDARY + OCR TABLE DETECTION")
    print("=" * 60)
    print("This enhanced method uses:")
    print("• Multi-method edge detection (Canny, Sobel, Laplacian, Morphological)")
    print("• High contrast boundary analysis")
    print("• Rectangular structure enhancement")
    print("• Advanced OCR validation with multiple configurations")
    print("• Comprehensive scoring system")
    print("Perfect for tables with ANY kind of visible boundaries!")

    # Detect tables using enhanced boundary detection + OCR validation
    all_table_areas = process_all_pages_boundary_ocr(pdf_path, dpi=300, save_debug=True)

    # Your specific table names
    table_names = ["credit_score", "active_products", "default", "guaranteed"]

    # Generate enhanced Camelot code
    generated_code = generate_camelot_code_from_boundaries(pdf_path, all_table_areas, table_names)

    print(f"\n{'='*60}")
    print("GENERATED ENHANCED CAMELOT CODE:")
    print(f"{'='*60}")
    print(generated_code)

    # Verify the detected tables with FULL DataFrame printing
    print(f"\n{'='*60}")
    print("EXTRACTING AND DISPLAYING ALL TABLES")
    print(f"{'='*60}")

    extracted_tables = verify_detected_tables(pdf_path, all_table_areas, print_full_tables=True)

    # Print enhanced extraction summary
    print_extraction_summary(extracted_tables)

    # Save tables to CSV files for inspection with enhanced naming
    print(f"\n{'='*60}")
    print("SAVING TABLES TO CSV WITH QUALITY METRICS")
    print(f"{'='*60}")
    saved_files = save_tables_to_text(extracted_tables)

    # Print final coordinates for easy copy-paste
    print(f"\n{'='*60}")
    print("ENHANCED TABLE COORDINATES SUMMARY:")
    print(f"{'='*60}")

    for page_num, areas in all_table_areas.items():
        for i, table_info in enumerate(areas):
            print(f"Page {page_num}, Table {i+1}: {table_info['area']}")
            print(f"  Scores - Confidence: {table_info['confidence']:.3f}, "
                  f"Boundary: {table_info['boundary_score']:.3f}, "
                  f"Contrast: {table_info['contrast_score']:.3f}")

    print(f"\n{'='*60}")
    print("ENHANCED DEBUG FILES CREATED:")
    print(f"{'='*60}")
    print("• debug_page_X_enhanced_boundaries.jpg - Main detection visualization")
    print("• debug_page_X_combined_edges.jpg - Combined edge detection")
    print("• debug_page_X_enhanced_edges.jpg - Enhanced rectangular structures")
    print("• debug_page_X_horizontal_lines.jpg - Horizontal line detection")
    print("• debug_page_X_vertical_lines.jpg - Vertical line detection")
    print("• debug_page_X_intersections.jpg - Line intersection detection")
    print("• debug_page_X_final_combined.jpg - Final processed image")
    print("\nColor coding:")
    print("• Red rectangles = All detected boundaries (with scores)")
    print("• Green rectangles = Validated tables (with detailed metrics)")

    # Performance summary
    total_detections = sum(len(areas) for areas in all_table_areas.values())
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY:")
    print(f"{'='*60}")
    print(f"Total pages processed: {len(all_table_areas)}")
    print(f"Total tables detected: {total_detections}")
    if total_detections > 0:
        avg_confidence = sum(
            table['confidence']
            for areas in all_table_areas.values()
            for table in areas
        ) / total_detections
        print(f"Average detection confidence: {avg_confidence:.3f}")

    print("\nEnhanced detection complete! Check the debug images to see the detection process.")
    return {
    "total_pages": len(all_table_areas),
    "tables_found": total_detections,
    "saved_files": saved_files,
}
