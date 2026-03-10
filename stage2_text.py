import os
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
index = faiss.IndexFlatL2(embedding_dim)
deepseek_llm = OllamaLLM(model="deepseek-r1:7b")

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text if text.strip() else None

# -----------------------------
# PDF Image Extraction
# -----------------------------
def extract_images_from_pdf(pdf_path, output_dir=image_output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index in range(len(doc)):
        page = doc[page_index]
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
            image_paths.append(image_path)

    return image_paths

# -----------------------------
# Text Chunking
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
# Store Text Chunks in FAISS
# -----------------------------
def store_in_faiss(text_data):
    vectors = []
    text_chunks = []

    for doc_id, text in text_data.items():
        if isinstance(text, str) and text.strip():
            chunks = chunk_text(text)
            text_chunks.extend(chunks)
            chunk_vectors = embedding_model.encode(chunks)
            vectors.extend(chunk_vectors)

    if vectors:
        vectors = np.array(vectors).astype("float32")
        index.add(vectors)

    return text_chunks

# -----------------------------
# Final Workflow Generation
# -----------------------------
def generate_workflow_with_deepseek(user_query, retrieved_text, image_paths):
    image_context = "\n".join(image_paths[:5]) if image_paths else "No figures extracted."

    prompt = (
        f"User Query:\n{user_query}\n\n"
        f"Retrieved Text from Research Papers:\n{retrieved_text}\n\n"
        f"Extracted Figures from the Papers (file paths):\n{image_context}\n\n"
        f"Use the user query and retrieved research evidence to generate a structured, step-by-step workflow "
        f"for solving the problem. If the extracted figures seem useful, mention how diagrams, charts, or visual "
        f"evidence from the papers could support implementation."
    )

    return deepseek_llm.invoke(prompt)

# -----------------------------
# Main
# -----------------------------
def main():
    user_query = "I want to build a Bio medical NER model using Bert"

    pdf_results = {}
    all_image_paths = []
    aggregated_text = ""

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Processing {pdf_file}...")

            extracted_text = extract_text_from_pdf(pdf_path)
            extracted_images = extract_images_from_pdf(pdf_path)

            all_image_paths.extend(extracted_images)

            if extracted_text:
                aggregated_text += extracted_text + "\n\n"
                pdf_results[pdf_file] = extracted_text

    print("PDF text and image extraction complete.")

    text_chunks = store_in_faiss(pdf_results)
    print("Text chunking and FAISS indexing complete.")

    query_vector = embedding_model.encode([user_query]).astype("float32")
    D, I = index.search(query_vector, k=5)
    print("Similarity retrieval complete.")

    retrieved_text = "\n".join(
        [text_chunks[idx] for idx in I[0] if 0 <= idx < len(text_chunks)]
    )

    print("Generating workflow with DeepSeek...")
    final_response = generate_workflow_with_deepseek(
        user_query=user_query,
        retrieved_text=retrieved_text,
        image_paths=all_image_paths
    )

    with open("extracted_pdf_data.json", "w", encoding="utf-8") as f:
        json.dump(pdf_results, f, indent=4)

    with open("extracted_image_paths.json", "w", encoding="utf-8") as f:
        json.dump(all_image_paths, f, indent=4)

    with open("generated_workflow.txt", "w", encoding="utf-8") as f:
        f.write(final_response)

    print("\nGenerated Workflow:\n")
    print(final_response)
    print("\nExtraction complete! Files saved:")
    print("- extracted_pdf_data.json")
    print("- extracted_image_paths.json")
    print("- generated_workflow.txt")

if __name__ == "__main__":
    main()
