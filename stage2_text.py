import os
import re
import json
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# -----------------------------
# Config
# -----------------------------
pdf_dir = "PDF_data"
image_output_dir = "extracted_images"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = embedding_model.get_sentence_embedding_dimension()

text_index = faiss.IndexFlatL2(embedding_dim)
caption_index = faiss.IndexFlatL2(embedding_dim)

deepseek_llm = OllamaLLM(model="deepseek-r1:7b")


# -----------------------------
# Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        text = page.get_text()
        pages_text.append(text if text else "")
    return pages_text


# -----------------------------
# Simple Caption Extraction
# -----------------------------
def extract_caption_from_page_text(page_text):
    """
    Very simple caption extraction:
    looks for lines starting with Figure / Fig.
    """
    lines = [line.strip() for line in page_text.split("\n") if line.strip()]
    captions = []

    for line in lines:
        if re.match(r"^(Figure|Fig\.?)\s*\d+", line, re.IGNORECASE):
            captions.append(line)

    return captions


# -----------------------------
# Image + Caption Extraction
# -----------------------------
def extract_images_and_captions_from_pdf(pdf_path, output_dir=image_output_dir):
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    figure_records = []

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_text = page.get_text()
        page_captions = extract_caption_from_page_text(page_text)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            image_filename = f"{pdf_name}_page{page_index + 1}_img{img_index + 1}.png"
            image_path = os.path.join(output_dir, image_filename)

            if pix.n < 5:
                pix.save(image_path)
            else:
                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                rgb_pix.save(image_path)
                rgb_pix = None

            pix = None

            # Best-effort caption assignment:
            # if captions exist on the page, assign one by order, else fallback
            caption = page_captions[img_index] if img_index < len(page_captions) else f"Figure on page {page_index + 1}"

            figure_records.append({
                "paper": pdf_name,
                "page": page_index + 1,
                "image_path": image_path,
                "caption": caption
            })

    return figure_records


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=525, overlap=250):
    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# -----------------------------
# Build Text FAISS Index
# -----------------------------
def store_text_chunks_in_faiss(text_data):
    text_chunks = []
    vectors = []

    for doc_id, full_text in text_data.items():
        if isinstance(full_text, str) and full_text.strip():
            chunks = chunk_text(full_text)
            text_chunks.extend(chunks)
            chunk_vectors = embedding_model.encode(chunks)
            vectors.extend(chunk_vectors)

    if vectors:
        vectors = np.array(vectors).astype("float32")
        text_index.add(vectors)

    return text_chunks


# -----------------------------
# Build Caption FAISS Index
# -----------------------------
def store_caption_embeddings_in_faiss(figure_records):
    captions = [record["caption"] for record in figure_records if record["caption"].strip()]

    if not captions:
        return []

    caption_vectors = embedding_model.encode(captions)
    caption_vectors = np.array(caption_vectors).astype("float32")
    caption_index.add(caption_vectors)

    return captions


# -----------------------------
# Retrieve Relevant Text Chunks
# -----------------------------
def retrieve_top_text_chunks(user_query, text_chunks, k=5):
    query_vector = embedding_model.encode([user_query]).astype("float32")
    D, I = text_index.search(query_vector, k)

    retrieved_chunks = [text_chunks[idx] for idx in I[0] if 0 <= idx < len(text_chunks)]
    return retrieved_chunks


# -----------------------------
# Retrieve Relevant Figures by Caption
# -----------------------------
def retrieve_top_figures_by_caption(user_query, figure_records, k=3):
    if len(figure_records) == 0:
        return []

    query_vector = embedding_model.encode([user_query]).astype("float32")
    D, I = caption_index.search(query_vector, min(k, len(figure_records)))

    top_figures = [figure_records[idx] for idx in I[0] if 0 <= idx < len(figure_records)]
    return top_figures


# -----------------------------
# Final Generation
# -----------------------------
def generate_workflow_with_deepseek(user_query, retrieved_text_chunks, top_figures):
    retrieved_text = "\n\n".join(retrieved_text_chunks)

    figure_context = ""
    if top_figures:
        for fig in top_figures:
            figure_context += (
                f"Paper: {fig['paper']}\n"
                f"Page: {fig['page']}\n"
                f"Caption: {fig['caption']}\n"
                f"Image Path: {fig['image_path']}\n\n"
            )
    else:
        figure_context = "No relevant figures found."

    prompt = (
        f"User Query:\n{user_query}\n\n"
        f"Retrieved Text Chunks from Research Papers:\n{retrieved_text}\n\n"
        f"Relevant Figures Retrieved Using Caption Similarity:\n{figure_context}\n"
        f"Using the retrieved research evidence, generate a structured, step-by-step workflow "
        f"to solve the user's problem. If the figure captions suggest useful architectural, methodological, "
        f"or experimental details, incorporate them into the workflow."
    )

    return deepseek_llm.invoke(prompt)


# -----------------------------
# Main
# -----------------------------
def main():
    user_query = "I want to build a Bio medical NER model using Bert"

    pdf_results = {}
    figure_records = []

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Processing {pdf_file}...")

            # Extract page-level text
            pages_text = extract_text_from_pdf(pdf_path)
            full_text = "\n\n".join(pages_text)

            if full_text.strip():
                pdf_results[pdf_file] = full_text

            # Extract images + captions
            figures = extract_images_and_captions_from_pdf(pdf_path)
            figure_records.extend(figures)

    print("PDF text and figure extraction complete.")

    # Build text chunk index
    text_chunks = store_text_chunks_in_faiss(pdf_results)
    print("Text chunk indexing complete.")

    # Build caption index
    captions = store_caption_embeddings_in_faiss(figure_records)
    print("Caption indexing complete.")

    # Retrieve top text chunks
    retrieved_text_chunks = retrieve_top_text_chunks(user_query, text_chunks, k=5)
    print("Relevant text chunks retrieved.")

    # Retrieve top figures using captions
    top_figures = retrieve_top_figures_by_caption(user_query, figure_records, k=3)
    print("Relevant figures retrieved using caption similarity.")

    # Generate final response
    final_response = generate_workflow_with_deepseek(user_query, retrieved_text_chunks, top_figures)
    print("Workflow generation complete.")

    # Save outputs
    with open("extracted_pdf_data.json", "w", encoding="utf-8") as f:
        json.dump(pdf_results, f, indent=4)

    with open("figure_records.json", "w", encoding="utf-8") as f:
        json.dump(figure_records, f, indent=4)

    with open("retrieved_text_chunks.json", "w", encoding="utf-8") as f:
        json.dump(retrieved_text_chunks, f, indent=4)

    with open("retrieved_figures.json", "w", encoding="utf-8") as f:
        json.dump(top_figures, f, indent=4)

    with open("generated_workflow.txt", "w", encoding="utf-8") as f:
        f.write(final_response)

    print("\nGenerated Workflow:\n")
    print(final_response)
    print("\nSaved:")
    print("- extracted_pdf_data.json")
    print("- figure_records.json")
    print("- retrieved_text_chunks.json")
    print("- retrieved_figures.json")
    print("- generated_workflow.txt")


if __name__ == "__main__":
    main()
