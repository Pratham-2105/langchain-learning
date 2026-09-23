from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

text = """
The Lockheed Martin F-16V Viper is a family of upgraded single-engine supersonic multirole fighter aircraft developed from the General Dynamics F-16 Fighting Falcon.Improvements over the older F-16 variants include the APG-83 AESA radar, upgraded avionics with a modern mission computer, enhanced cockpit displays, and electronic warfare suites such as the AN/ALQ-254(V)1 Viper Shield. According to the manufacturer, the aircraft delivers fifth-generation fighter radar capabilities by leveraging hardware and software commonality with F-22 and F-35 AESA radars while extending service life and interoperability with stealth platforms.[4][5][6] Initially introduced to support export customers like Taiwan and Greece through both new production and retrofit options, the Viper configuration continues to be fielded and modernized by Lockheed Martin and international partners.[7][8] In the United States Air Force, the F-16 C/D Post Block Integration Team (PoBIT) modernization program is equivalent.[9]

Development
The F-16V (Block 70/72) began as Lockheed Martin’s comprehensive "Viper" upgrade announced at the Singapore Airshow on February 15, 2012, adding the AN/APG-83 SABR AESA radar, a new mission computer/architecture, and cockpit enhancements.An F-16V achieved its maiden flight with the AN/APG-83 on October 16, 2015, validating the configuration that would underpin both new-build Block 70/72 jets and global retrofit programs.
"""

loader = PyPDFLoader('10.TextSplitters/nn.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

#text_result = splitter.split_text(text)
#print(text_result)

document_result = splitter.split_documents(docs)
print(document_result)
print(document_result[0])
