from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('9.DL/books/nn.pdf')

docs = loader.load()

print(docs[5])
print(docs[5].page_content)
print(docs[5].metadata)
#print(docs)
#print(len(docs))