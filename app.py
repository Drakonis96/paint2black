import os
import uuid
import threading
import time
import json
import tempfile # <--- Añadir import
from flask import (
    Flask, request, render_template, send_file,
    after_this_request, abort, flash, redirect, url_for,
    jsonify, Response
)
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from io import BytesIO
# from PyPDF2 import PdfMerger # <--- Ya no se necesita para OCR
import cv2
import numpy as np
import concurrent.futures
import logging

# --- Configuration (sin cambios) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['SUPPORTED_OCR_LANGUAGES'] = ['eng', 'spa', 'fra', 'deu']
app.config['CONVERSION_DPI'] = 300

# --- Task State Management (sin cambios) ---
tasks_progress = {}
tasks_lock = threading.Lock()

# --- Directory Setup (sin cambios) ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try: os.makedirs(app.config['UPLOAD_FOLDER'])
    except OSError as e: logger.error(f"Fatal: Could not create temp directory {app.config['UPLOAD_FOLDER']}: {e}"); exit(1)

# --- Helper Functions (sin cambios) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def update_task_progress(task_id, current, total, status="processing", message=None, result_file_id=None, result_suffix=None):
    with tasks_lock:
        if task_id not in tasks_progress: tasks_progress[task_id] = {}
        tasks_progress[task_id].update({
            'current': current, 'total': total, 'status': status,
            'message': message if message is not None else tasks_progress[task_id].get('message', ''),
            'result_file_id': result_file_id if result_file_id is not None else tasks_progress[task_id].get('result_file_id'),
            'result_suffix': result_suffix if result_suffix is not None else tasks_progress[task_id].get('result_suffix'),
            'last_update': time.time()
        })

def get_task_progress(task_id):
    with tasks_lock: return tasks_progress.get(task_id, {}).copy()

def cleanup_file(file_path, task_id, file_description):
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Task {task_id}: Cleaned up {file_description} file '{os.path.basename(file_path)}'.")
        except OSError as e:
            logger.error(f"Task {task_id}: Error cleaning up {file_description} file '{os.path.basename(file_path)}': {e}")

# --- === STEP 1: Color Conversion Task (sin cambios) === ---
def run_conversion_task(task_id, original_path, converted_pdf_path, mode):
    # ... (código idéntico a la versión anterior) ...
    total_pages = 0
    start_time = time.time()
    result_file_id = task_id
    output_doc = None
    doc = None
    try:
        update_task_progress(task_id, 0, 0, status="starting", message="Opening PDF...")
        doc = fitz.open(original_path)
        total_pages = len(doc)
        if total_pages == 0: raise ValueError("PDF contains no pages.")
        update_task_progress(task_id, 0, total_pages, status="processing", message=f"Converting {total_pages} pages...")
        output_doc = fitz.open()
        dpi = app.config['CONVERSION_DPI']
        zoom_x = dpi / 72.0; zoom_y = dpi / 72.0
        mat = fitz.Matrix(zoom_x, zoom_y)

        for i, page in enumerate(doc):
            current_page_num = i + 1
            update_task_progress(task_id, current_page_num, total_pages, status="processing", message=f"Processing page {current_page_num}/{total_pages}...")
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if img_np.shape[2] == 3: gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            elif img_np.shape[2] == 4: gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
            else: gray = img_np
            if mode == 'monochrome': _, processed_img_np = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else: processed_img_np = gray
            is_success, buffer = cv2.imencode(".png", processed_img_np)
            if not is_success: raise ValueError(f"Failed to encode processed image for page {current_page_num}")
            img_bytes = buffer.tobytes()
            page_rect = page.rect
            new_page = output_doc.new_page(width=page_rect.width, height=page_rect.height)
            new_page.insert_image(page_rect, stream=img_bytes)

        update_task_progress(task_id, total_pages, total_pages, status="saving", message="Saving converted PDF...")
        output_doc.save(converted_pdf_path, garbage=4, deflate=True)
        output_doc.close(); output_doc = None # Mark as closed
        doc.close(); doc = None # Mark as closed
        end_time = time.time(); duration = end_time - start_time
        logger.info(f"Task {task_id}: Color conversion completed in {duration:.2f}s. Saved to '{converted_pdf_path}'.")
        update_task_progress(task_id, total_pages, total_pages, status="complete", message=f"Color conversion complete ({total_pages} pages). Ready for download.", result_file_id=result_file_id, result_suffix="_converted.pdf")
    except Exception as e:
        logger.error(f"Task {task_id}: Error during color conversion: {e}", exc_info=True)
        update_task_progress(task_id, tasks_progress.get(task_id, {}).get('current', 0), total_pages, status="error", message=f"Conversion Error: {e}")
    finally:
        if output_doc and output_doc.is_open: output_doc.close()
        if doc and doc.is_open: doc.close()
        cleanup_file(original_path, task_id, "original (conversion)")

# --- === STEP 2: OCR Processing Task (MODIFICADO) === ---

def process_page_ocr_only(image_np, lang_code, temp_dir, task_id, page_num):
    """
    Aplica OCR a una imagen NumPy y GUARDA el resultado en un PDF temporal de 1 página.
    Devuelve la RUTA al archivo temporal o None si falla.
    """
    temp_pdf_path = None
    try:
        # Preprocesamiento (igual que antes)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            processed_img_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            processed_img_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
        else:
            processed_img_np = image_np

        custom_config = r'--oem 1 --psm 3'
        # Generar PDF con texto incrustado
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            processed_img_np, extension='pdf', lang=lang_code, config=custom_config
        )

        if not pdf_bytes:
             logger.warning(f"Task {task_id}, Page {page_num+1}: pytesseract returned empty bytes.")
             return None

        # Crear un archivo temporal seguro para guardar esta página
        # Usamos el directorio de trabajo temporal específico de la tarea
        fd, temp_pdf_path = tempfile.mkstemp(suffix=f"_page{page_num}.pdf", prefix=f"{task_id}_", dir=temp_dir)
        os.close(fd) # Cerramos el descriptor de archivo, solo necesitamos el path

        with open(temp_pdf_path, "wb") as f_temp:
            f_temp.write(pdf_bytes)

        # logger.debug(f"Task {task_id}, Page {page_num+1}: Saved temporary OCR PDF to {temp_pdf_path}")
        return temp_pdf_path # Devolver la ruta al archivo guardado

    except pytesseract.TesseractNotFoundError:
        logger.error(f"Task {task_id}: Tesseract not found in worker process!")
        raise # Relanzar para que el proceso principal se entere
    except cv2.error as cv_err:
        logger.error(f"Task {task_id}, Page {page_num+1}: OpenCV error during OCR pre-processing: {cv_err}")
        if temp_pdf_path and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path) # Clean up temp file on error
        return None
    except Exception as e:
        logger.error(f"Task {task_id}, Page {page_num+1}: Error processing page for OCR: {e}", exc_info=True)
        if temp_pdf_path and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path) # Clean up temp file on error
        return None


def run_ocr_task(task_id, input_pdf_path, ocr_pdf_path, lang_code):
    """Background thread function for OCR processing using PyMuPDF for merging."""
    total_pages = 0
    start_time = time.time()
    result_file_id = task_id
    images_np = []
    temp_page_files = [] # Lista para guardar rutas de archivos temporales por página
    final_doc = None
    doc = None
    ocr_temp_dir = None # Directorio temporal para los PDFs de página única

    try:
        # Crear un directorio temporal específico para los archivos intermedios de esta tarea OCR
        # Usar tempfile.TemporaryDirectory asegura que se limpie automáticamente (en teoría)
        # pero haremos limpieza manual explícita por si acaso.
        ocr_temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"ocr_{task_id}_", dir=app.config['UPLOAD_FOLDER'])
        ocr_temp_dir = ocr_temp_dir_obj.name
        logger.info(f"Task {task_id}: Created temporary OCR directory: {ocr_temp_dir}")


        # 1. Extract images from the input PDF
        update_task_progress(task_id, 0, 0, status="starting", message="Extracting pages for OCR...")
        doc = fitz.open(input_pdf_path)
        total_pages = len(doc)
        if total_pages == 0: raise ValueError("Input PDF for OCR contains no pages.")
        update_task_progress(task_id, 0, total_pages, status="extracting", message=f"Extracting {total_pages} pages...")

        dpi = app.config['CONVERSION_DPI']
        zoom_x = dpi / 72.0; zoom_y = dpi / 72.0
        mat = fitz.Matrix(zoom_x, zoom_y)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images_np.append(np.array(img))
        doc.close(); doc = None # Cerrar tan pronto como sea posible
        logger.info(f"Task {task_id}: Image extraction complete ({total_pages} pages).")

        # 2. Perform OCR in Parallel - MODIFICADO para guardar archivos temporales
        update_task_progress(task_id, 0, total_pages, status="processing", message="Starting OCR processing...")
        max_ocr_workers = max(1, (os.cpu_count() or 1) // 2)
        logger.info(f"Task {task_id}: Using ProcessPoolExecutor for OCR with max_workers={max_ocr_workers}")

        # Usaremos un diccionario para mantener el orden correcto de las páginas
        page_results_paths = {} # {page_index: path_or_None}
        processed_count = 0
        tesseract_error_flag = False

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_ocr_workers) as executor:
            future_to_page = {
                # Pasar el directorio temporal ocr_temp_dir a la función del worker
                executor.submit(process_page_ocr_only, img_np, lang_code, ocr_temp_dir, task_id, i): i
                for i, img_np in enumerate(images_np)
            }
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    # El resultado ahora es la RUTA al archivo temporal o None
                    temp_pdf_path = future.result()
                    page_results_paths[page_num] = temp_pdf_path # Guardar ruta o None
                    if temp_pdf_path:
                        temp_page_files.append(temp_pdf_path) # Añadir a lista general para limpieza
                except pytesseract.TesseractNotFoundError:
                    tesseract_error_flag = True
                    logger.error(f"Task {task_id}: Tesseract not found during parallel execution!")
                    for f in future_to_page: f.cancel()
                    break
                except Exception as exc:
                    logger.error(f'Task {task_id}: Page {page_num + 1} OCR generated exception in pool: {exc}', exc_info=True)
                    page_results_paths[page_num] = None # Marcar como fallida

                processed_count += 1
                update_task_progress(task_id, processed_count, total_pages, status="processing", message=f"Running OCR on page {processed_count}/{total_pages}...")
                if tesseract_error_flag: break

        if tesseract_error_flag:
            raise pytesseract.TesseractNotFoundError("Tesseract not found during processing.")

        # 3. Merge OCR results using PyMuPDF - MODIFICADO
        update_task_progress(task_id, processed_count, total_pages, status="merging", message="Merging OCR results...")
        final_doc = fitz.open() # Documento PDF final vacío
        pages_added = 0

        # Iterar en el orden correcto de las páginas
        for i in range(total_pages):
            temp_path = page_results_paths.get(i)
            if temp_path and os.path.exists(temp_path):
                try:
                    page_doc = fitz.open(temp_path)
                    final_doc.insert_pdf(page_doc) # Inserta el contenido de la página
                    page_doc.close()
                    pages_added += 1
                except Exception as e:
                    logger.error(f"Task {task_id}: Error inserting page {i+1} from '{temp_path}' using Fitz: {e}")
                    if 'page_doc' in locals() and page_doc.is_open: page_doc.close()
            else:
                 logger.warning(f"Task {task_id}: Skipping page {i+1} in merge (no result file found or OCR failed).")


        if pages_added == 0:
            raise ValueError("No pages were successfully processed by OCR to merge.")

        # 4. Save the final OCR'd PDF
        final_doc.save(ocr_pdf_path, garbage=4, deflate=True) # Guardar el documento final
        final_doc.close(); final_doc = None # Cerrar y marcar como cerrado
        logger.info(f"Task {task_id}: Successfully merged {pages_added} pages using PyMuPDF.")

        end_time = time.time(); duration = end_time - start_time
        logger.info(f"Task {task_id}: OCR processing completed in {duration:.2f}s. Saved to '{ocr_pdf_path}'.")
        final_message = f"OCR processing complete ({pages_added}/{total_pages} pages)."
        if pages_added < total_pages: final_message += " Some pages may have failed."
        update_task_progress(task_id, pages_added, total_pages, status="complete", message=final_message, result_file_id=result_file_id, result_suffix="_ocr.pdf")

    except pytesseract.TesseractNotFoundError as e:
         update_task_progress(task_id, tasks_progress.get(task_id, {}).get('current', 0), total_pages, status="error", message=f"OCR Error: {e}")
    except Exception as e:
        logger.error(f"Task {task_id}: Error during OCR task: {e}", exc_info=True)
        update_task_progress(task_id, tasks_progress.get(task_id, {}).get('current', 0), total_pages, status="error", message=f"OCR Error: {e}")
    finally:
        # Cerrar documentos si aún están abiertos
        if final_doc and final_doc.is_open: final_doc.close()
        if doc and doc.is_open: doc.close()
        # Limpiar archivo de entrada para OCR
        cleanup_file(input_pdf_path, task_id, "input (OCR)")
        # Limpiar archivos temporales de páginas individuales
        logger.info(f"Task {task_id}: Cleaning up temporary page files...")
        for temp_file in temp_page_files:
            cleanup_file(temp_file, task_id, "temporary page")
        # Limpiar el directorio temporal (si se usó el objeto)
        if 'ocr_temp_dir_obj' in locals():
             try:
                 ocr_temp_dir_obj.cleanup()
                 logger.info(f"Task {task_id}: Cleaned up temporary directory {ocr_temp_dir}")
             except Exception as e:
                 logger.error(f"Task {task_id}: Error cleaning up temporary directory {ocr_temp_dir}: {e}")
        # O limpiar manualmente si solo se usó el path (menos seguro)
        # elif ocr_temp_dir and os.path.exists(ocr_temp_dir):
        #    try: shutil.rmtree(ocr_temp_dir) # Requires shutil import
        #    except Exception as e: logger.error(f"...")


# --- === Flask Routes (sin cambios) === ---
@app.route('/')
def index():
    return render_template('index.html', supported_languages=app.config['SUPPORTED_OCR_LANGUAGES'])

@app.route('/convert_color', methods=['POST'])
def convert_color_request():
    if 'pdf_file' not in request.files: return jsonify({'error': 'No PDF file part in the request.'}), 400
    file = request.files['pdf_file']
    if file.filename == '' or not file or not allowed_file(file.filename): return jsonify({'error': 'No selected file or file type not allowed.'}), 400
    mode = request.form.get('mode', 'monochrome')
    if mode not in ['monochrome', 'grayscale']: return jsonify({'error': 'Invalid mode selected.'}), 400
    task_id = str(uuid.uuid4())
    original_filename = secure_filename(file.filename)
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_orig_conv.pdf")
    converted_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_converted.pdf")
    try:
        file.save(original_path)
        logger.info(f"Task {task_id}: Saved original for conversion: '{original_filename}' -> '{os.path.basename(original_path)}'")
        update_task_progress(task_id, 0, 0, status="queued", message="Color conversion task queued.")
        thread = threading.Thread(target=run_conversion_task, args=(task_id, original_path, converted_pdf_path, mode))
        thread.daemon = True; thread.start()
        return jsonify({'task_id': task_id})
    except Exception as e:
        logger.error(f"Task {task_id}: Failed to start conversion task: {e}", exc_info=True)
        cleanup_file(original_path, task_id, "original (failed start)")
        return jsonify({'error': f'Failed to start conversion: {e}'}), 500

@app.route('/perform_ocr', methods=['POST'])
def perform_ocr_request():
    if 'pdf_file_ocr' not in request.files: return jsonify({'error': 'No PDF file part in the request for OCR.'}), 400
    file = request.files['pdf_file_ocr']
    if file.filename == '' or not file or not allowed_file(file.filename): return jsonify({'error': 'No selected file or file type not allowed for OCR.'}), 400
    lang_code = request.form.get('lang', 'eng')
    if app.config['SUPPORTED_OCR_LANGUAGES'] and lang_code not in app.config['SUPPORTED_OCR_LANGUAGES']: lang_code = 'eng'
    task_id = str(uuid.uuid4())
    original_filename = secure_filename(file.filename)
    input_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_orig_ocr.pdf")
    ocr_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_ocr.pdf")
    try:
        file.save(input_pdf_path)
        logger.info(f"Task {task_id}: Saved input for OCR: '{original_filename}' -> '{os.path.basename(input_pdf_path)}'")
        update_task_progress(task_id, 0, 0, status="queued", message="OCR task queued.")
        thread = threading.Thread(target=run_ocr_task, args=(task_id, input_pdf_path, ocr_pdf_path, lang_code))
        thread.daemon = True; thread.start()
        return jsonify({'task_id': task_id})
    except Exception as e:
        logger.error(f"Task {task_id}: Failed to start OCR task: {e}", exc_info=True)
        cleanup_file(input_pdf_path, task_id, "input (failed start)")
        return jsonify({'error': f'Failed to start OCR: {e}'}), 500

@app.route('/stream/<task_id>')
def stream_progress(task_id):
    # ... (código idéntico a la versión anterior) ...
    def generate():
        last_sent_state = None; logger.info(f"SSE stream opened for Task {task_id}")
        while True:
            progress = get_task_progress(task_id)
            current_state = (progress.get('current'), progress.get('total'), progress.get('status'), progress.get('message'), progress.get('result_file_id'), progress.get('result_suffix'))
            if not progress:
                data = {'status': 'error', 'message': 'Task not found or expired.'}; yield f"data: {json.dumps(data)}\n\n"; logger.warning(f"SSE stream for Task {task_id}: Task data not found."); break
            if current_state != last_sent_state: yield f"data: {json.dumps(progress)}\n\n"; last_sent_state = current_state
            if progress.get('status') in ['complete', 'error']: logger.info(f"SSE stream closing for Task {task_id}. Final status: {progress.get('status')}"); break
            time.sleep(1)
        logger.info(f"SSE stream connection closed for Task {task_id}")
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<file_id>')
def download_result(file_id):
    # ... (código idéntico a la versión anterior) ...
    if not file_id or not all(c.isalnum() or c == '-' for c in file_id): abort(404, description="Invalid file/task ID.")
    task_id = file_id; progress = get_task_progress(task_id)
    if progress.get('status') != 'complete' or not progress.get('result_file_id') or not progress.get('result_suffix'):
        logger.warning(f"Download attempt for incomplete/invalid task {task_id}. Progress: {progress}"); abort(404, description="Task not complete or result file information missing.")
    result_file_base_id = progress['result_file_id']; result_suffix = progress['result_suffix']
    result_filename = f"{secure_filename(result_file_base_id)}{result_suffix}"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    logger.info(f"Download request for Task {task_id}, expecting file: {result_path}")
    if not os.path.exists(result_path): logger.error(f"Result file not found for task {task_id}: {result_path}"); abort(404, description="Result file not found. It might have been cleaned up already.")
    @after_this_request
    def remove_file_after_download(response):
        cleanup_file(result_path, task_id, "result")
        with tasks_lock:
            if task_id in tasks_progress: del tasks_progress[task_id]; logger.info(f"Task {task_id}: Removed progress state from memory after download.")
        return response
    try:
        download_name_prefix = "converted" if result_suffix == "_converted.pdf" else "ocr_result"
        suggested_download_name = f"{download_name_prefix}_{result_file_base_id[:8]}.pdf"
        return send_file(result_path, as_attachment=True, download_name=suggested_download_name, mimetype='application/pdf')
    except Exception as e: logger.error(f"Error sending file '{result_path}' (Task {task_id}): {e}", exc_info=True); abort(500, description="Server error preparing file for download.")

# --- Error Handlers (sin cambios) ---
@app.errorhandler(404)
def page_not_found_error(e):
    logger.warning(f"404 Not Found: {request.url} - {e}"); desc = getattr(e, 'description', 'The requested URL was not found on the server.')
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html: return jsonify(error=desc), 404
    return render_template('404.html', error_description=desc), 404
@app.errorhandler(413)
def request_entity_too_large_error(e):
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / 1024 / 1024; logger.warning(f"413 Payload Too Large (Limit: {max_size_mb:.0f} MB)."); error_msg = f"File is too large. Maximum size is {max_size_mb:.0f} MB."
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html: return jsonify(error=error_msg), 413
    flash(error_msg, 'error'); return redirect(url_for('index'))
@app.errorhandler(500)
def internal_server_error_handler(e):
    logger.error(f"500 Internal Server Error: {e}", exc_info=True)
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html: return jsonify(error="An internal server error occurred."), 500
    return render_template('500.html'), 500

# --- Main Execution (sin cambios) ---
if __name__ == '__main__':
    print("--- Starting Flask Development Server (1 worker, threaded) ---"); app.run(host='0.0.0.0', port=5018, debug=False, threaded=True)