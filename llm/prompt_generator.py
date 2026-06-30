from langchain_core.prompts import PromptTemplate
import json


class PromptGenerator:
    # Template is built once at class level — same for every instance/paper.
    paper_explanation_prompt = PromptTemplate(
        input_variables=[
            "title",
            "authors",
            "abstract",
            "paper_link",
            "explanation_type_instruction",
            "explanation_length_instruction",
        ],
        template="""You are an expert research assistant who helps users understand academic papers clearly and accurately based on the information available.

Below is the information about a research paper retrieved from Google Scholar:

Title: {title}
Authors: {authors}
Abstract: {abstract}
Paper Link: {paper_link}

Your task is to generate an explanation of this paper for the user based on the following requirements:

1. Style of explanation: {explanation_type_instruction}
2. Length of explanation: {explanation_length_instruction}

Important guidelines:
- Give the output in markdown format, using LaTeX-style notation for any mathematical expressions.
-Kindly go through the publication link provided to you and get the knowledge of the paper in detail. 
- Base your explanation only on the paper detail provided to you. Do not fabricate specific results, numbers, or claims that are not mention in the papers.
- If you lack detailed information, do not make assumptions or invent content. Kindly acknowledge the limitations.
- Do not repeat the abstract verbatim — rephrase and explain it in your own words according to the requested style.
- Maintain a tone appropriate to the chosen style.
- Do not include a title heading, disclaimers, or meta-comments like "Here is the explanation" — output only the explanation itself.

Now write the explanation:""",
    )

    EXPLANATION_TYPE_INSTRUCTIONS = {
        "mathematics_oriented": (
            "Focus heavily on the mathematical formulation of the paper. "
            "Explain the key equations, models, theorems, or algorithms used. "
            "If the abstract references specific methods (e.g., optimization, "
            "statistical models, proofs), describe their mathematical intuition "
            "and notation as clearly as possible, even if some details must be "
            "inferred since only the abstract is available. Use LaTeX-style "
            "notation where appropriate."
        ),
        "beginner_friendly": (
            "Explain the paper as if to someone with no prior background in "
            "this field. Avoid jargon, or define it immediately in simple terms. "
            "Use analogies and everyday examples to make the core idea, "
            "motivation, and contribution of the paper intuitive and easy to grasp."
        ),
        "summary": (
            "Provide a concise, neutral, objective summary of the paper's "
            "purpose, methodology, and findings, written in clear academic "
            "language. Avoid unnecessary elaboration or simplification — focus "
            "on conveying the core contribution efficiently."
        ),
    }

    EXPLANATION_LENGTH_INSTRUCTIONS = {
        "short": "Keep the explanation to about 2-3 sentences (around 50-70 words).",
        "medium": "Keep the explanation to about 1-2 short paragraphs (around 150-200 words).",
        "long": "Provide a detailed explanation of about 3-5 paragraphs (around 350-450 words), covering context, method, and significance.",
    }

    def __init__(self, paper_info: dict, llm_settings: dict):
        self.paper_info = paper_info
        self.llm_settings = llm_settings

    def build_explanation_prompt(self) -> str:
        type_instr = self.EXPLANATION_TYPE_INSTRUCTIONS.get(
            self.llm_settings['explanation_type'],
            self.EXPLANATION_TYPE_INSTRUCTIONS["summary"]
        )
        length_instr = self.EXPLANATION_LENGTH_INSTRUCTIONS.get(
            self.llm_settings['explanation_length'],
            self.EXPLANATION_LENGTH_INSTRUCTIONS["medium"]
        )

        authors = self.paper_info['authors']
        if isinstance(authors, (list, tuple, set)):
            authors = ", ".join(authors)

        formatted = self.paper_explanation_prompt.format(
            title=self.paper_info['title'],
            authors=authors,
            abstract=self.paper_info['abstract'],
            paper_link=self.paper_info['link'],
            explanation_type_instruction=type_instr,
            explanation_length_instruction=length_instr,
        )

        with open("prompt_template.json", "w") as f:
            json.dump({
                "title": self.paper_info['title'],
                "authors": authors,
                "abstract": self.paper_info['abstract'],
                "paper_link": self.paper_info['link'],
                "explanation_type_instruction": type_instr,
                "explanation_length_instruction": length_instr,
            }, f, indent=4)

        return formatted


if __name__ == "__main__":
    paper_info = {
        "title": "Example Paper Title",
        "authors": ["Author A", "Author B"],
        "abstract": "This is an example abstract of the paper.",
        "link": "http://example.com/paper"
    }
    llm_settings = {
        "explanation_type": "mathematics_oriented",
        "explanation_length": "long"
    }

    prompt_generator = PromptGenerator(paper_info, llm_settings)
    final_prompt = prompt_generator.build_explanation_prompt()
    print(final_prompt)