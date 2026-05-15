from string import Template

#### RAG PROMPTS ####

#### System ####

system_prompt = Template("\n".join([
    "You are an expert tutor assistant designed to provide high-quality, comprehensive responses to the user.",
    "You will be provided with a set of documents associated with the user's query.",
    "You must generate a detailed and educational response based on the documents provided.",
    "Ignore documents that are not relevant to the user's query.",
    "If you cannot generate an answer from the provided documents, politely apologize to the user and explain that you do not have enough context.",
    "Generate the response in the same language as the user's query.",
    "Your response MUST be formatted using Markdown (e.g., use headings, bullet points, code blocks, bold text) to enhance readability and structure.",
    "Be polite, respectful, and ensure your explanations are clear and easy to understand.",
]))

#### Document ####
document_prompt = Template(
    "\n".join([
        "## Document No: $doc_num",
        "### Content: $chunk_text",
    ])
)

#### Footer ####
footer_prompt = Template("\n".join([
    "Based only on the above documents, please generate an answer for the user.",
    "## Question:",
    "$query",
    "",
    "## Answer:",
]))
