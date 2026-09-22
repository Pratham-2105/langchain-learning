import gc
from time import perf_counter

import psutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# ==========================================================
# PROCESS / MEMORY HELPERS
# ==========================================================

process = psutil.Process()


def memory_mb():
    """Return current process RSS memory in MB."""
    return process.memory_info().rss / (1024 * 1024)


# ==========================================================
# LOADER FACTORY
# ==========================================================

def create_loader():
    return DirectoryLoader(
        path="9.DL/books",
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )


# ==========================================================
# 1. NORMAL LOAD()
# ==========================================================

print("\n==========================================")
print("              NORMAL LOAD()")
print("==========================================")

gc.collect()

before_normal_memory = memory_mb()

loader = create_loader()

start = perf_counter()

docs = loader.load()

normal_time = perf_counter() - start

after_normal_memory = memory_mb()

print(f"Time taken        : {normal_time:.4f} seconds")
print(f"Documents         : {len(docs)}")
print(f"Memory before     : {before_normal_memory:.2f} MB")
print(f"Memory after      : {after_normal_memory:.2f} MB")
print(
    f"Memory increase   : "
    f"{after_normal_memory - before_normal_memory:.2f} MB"
)


# ==========================================================
# SAMPLE DOCUMENT
# ==========================================================

print("\n========== SAMPLE DOCUMENT ==========")

print(f"Source : {docs[0].metadata.get('source')}")
print(f"Page   : {docs[0].metadata.get('page')}")
print(f"Text   : {docs[0].page_content[:150]}...")


# ==========================================================
# FREE NORMAL DOCUMENTS
# ==========================================================

del docs
gc.collect()

print("\n========== AFTER NORMAL LOAD ==========")
print(f"Memory after deleting docs : {memory_mb():.2f} MB")


# ==========================================================
# 2. LAZY LOAD() — CREATE ITERATOR
# ==========================================================

print("\n==========================================")
print("            LAZY LOAD()")
print("==========================================")

gc.collect()

loader = create_loader()

before_lazy_memory = memory_mb()

start = perf_counter()

lazy_docs = loader.lazy_load()

iterator_time = perf_counter() - start

after_iterator_memory = memory_mb()

print(f"Iterator creation time : {iterator_time:.6f} seconds")
print(f"Memory before iterator : {before_lazy_memory:.2f} MB")
print(f"Memory after iterator  : {after_iterator_memory:.2f} MB")


# ==========================================================
# 3. FIRST DOCUMENT
# ==========================================================

print("\n========== FIRST DOCUMENT ==========")

start = perf_counter()

first_doc = next(lazy_docs)

first_doc_time = perf_counter() - start

print(f"Time taken : {first_doc_time:.4f} seconds")
print(f"Source     : {first_doc.metadata.get('source')}")
print(f"Page       : {first_doc.metadata.get('page')}")

del first_doc


# ==========================================================
# 4. SECOND DOCUMENT
# ==========================================================

print("\n========== SECOND DOCUMENT ==========")

start = perf_counter()

second_doc = next(lazy_docs)

second_doc_time = perf_counter() - start

print(f"Time taken : {second_doc_time:.4f} seconds")
print(f"Source     : {second_doc.metadata.get('source')}")
print(f"Page       : {second_doc.metadata.get('page')}")

del second_doc


# ==========================================================
# 5. CONSUME EVERY LAZY DOCUMENT
# ==========================================================

print("\n==========================================")
print("       CONSUMING ALL LAZY DOCUMENTS")
print("==========================================")

# Start from the beginning again.
loader = create_loader()

gc.collect()

lazy_start_memory = memory_mb()
peak_memory = lazy_start_memory

start = perf_counter()

count = 0

for doc in loader.lazy_load():

    # ------------------------------------------------------
    # Pretend we're processing the document.
    #
    # Example:
    #
    # text = doc.page_content
    # source = doc.metadata.get("source")
    #
    # We DO NOT append doc to a list.
    # ------------------------------------------------------

    count += 1

    current_memory = memory_mb()

    if current_memory > peak_memory:
        peak_memory = current_memory

lazy_total_time = perf_counter() - start

final_memory = memory_mb()


# ==========================================================
# 6. FINAL COMPARISON
# ==========================================================

print("\n==========================================")
print("              FINAL COMPARISON")
print("==========================================")

print(f"Normal load time          : {normal_time:.4f} seconds")
print(f"Lazy full-consumption time: {lazy_total_time:.4f} seconds")

print(f"\nNormal load memory increase:")
print(
    f"  {after_normal_memory - before_normal_memory:.2f} MB"
)

print(f"\nLazy loading starting memory:")
print(
    f"  {lazy_start_memory:.2f} MB"
)

print(f"Lazy loading peak memory:")
print(
    f"  {peak_memory:.2f} MB"
)

print(f"Lazy loading final memory:")
print(
    f"  {final_memory:.2f} MB"
)

print(f"\nLazy documents consumed: {count}")


# ==========================================================
# 7. INTERPRETATION
# ==========================================================

print("\n==========================================")
print("             WHAT THIS SHOWS")
print("==========================================")

print("""
load():
    - Parses all PDFs immediately.
    - Creates all Document objects.
    - Keeps the entire list in memory.
    - You pay the loading cost immediately.

lazy_load():
    - Creates an iterator almost instantly.
    - Documents are produced as the iterator is consumed.
    - Previous documents are not retained by our loop.
    - Total parsing work still has to happen if we consume everything.
""")