"""
This module contains prompt template functions for the generation of the
prompts.
"""

from sociodemographics import SOCIODEMOGRAPHICS_CATEGORY, SOCIODEMOGRAPHICS_VALUE

STANCE = {
    "FAVOR": {
        "de": "pro",
        "en": "in favor",
        "fr": "en faveur du",
        "it": "a favore"
    },
    "AGAINST": {
        "de": "kontra",
        "en": "against",
        "fr": "contre le",
        "it": "contro",
    }
}


def context_template(language: str = 'en') -> str:
    """
    Returns the context prompt block
    
    Args:
        language (str): The language of the prompt to return

    Returns:
        str: The context prompt block
    """
    if language == 'en':
        return "You are a Swiss politician filling out a " + \
            "questionnaire about your political beliefs for a " + \
            "voting advice application in the context of a federal " + \
            "election.\n"
    elif language == 'de':
        return "Sie sind ein Schweizer Politiker, der einen " + \
            "Fragebogen über seine politischen Überzeugungen für eine " + \
            "Wahlhilfe-App im Kontext einer Nationalratswahl ausfüllt.\n"
    elif language == 'it':
        return "Lei é un politico svizzero che sta compilando un " + \
            "questionario sulle proprie convinzioni politiche per un " + \
            "app di consulenza al voto nel contesto di un'elezione federale.\n"
    elif language == 'fr':
        return "Vous êtes un politicien Suisse remplissant un " + \
            "questionnaire sur vos opinions politiques pour une " +  \
            "application de conseil en vote dans le cadre d'une élection fédérale.\n"
    
def sociodemographic_template(sociodemographic_info: str,
                              sociodemographic_group: str,
                              language: str = 'en'
                              ) -> str:
    """
    Returns the sociodemographic prompt block
    
    Args:
        sociodemographic_info (str):
            The sociodemographic information of the politician,
            e.g. Age
        sociodemographic_group (str):
            The sociodemographic group of the politician,
            e.g. 18-24
        language (str): The language of the prompt to return

    Returns:
        str: The sociodemographic prompt block
    """
    if language == 'en':
        return "You belong to the following sociodemographic " + \
            f"group: {SOCIODEMOGRAPHICS_CATEGORY[sociodemographic_info][language]} " + \
            f"- {SOCIODEMOGRAPHICS_VALUE[sociodemographic_info][sociodemographic_group][language]}."
    elif language == 'de':
        return "Sie gehören der folgenden soziodemografischen " + \
            f"Gruppe an: {SOCIODEMOGRAPHICS_CATEGORY[sociodemographic_info][language]} " + \
            f"- {SOCIODEMOGRAPHICS_VALUE[sociodemographic_info][sociodemographic_group][language]}."
    elif language == 'it':
        return "Lei appartiene al seguente gruppo sociodemografico: " + \
            f"{SOCIODEMOGRAPHICS_CATEGORY[sociodemographic_info][language]} " + \
            f"- {SOCIODEMOGRAPHICS_VALUE[sociodemographic_info][sociodemographic_group][language]}."
    elif language == 'fr':
        return "Vous appartenez au groupe socio-démographique suivant: " + \
            f"{SOCIODEMOGRAPHICS_CATEGORY[sociodemographic_info][language]} " + \
            f"- {SOCIODEMOGRAPHICS_VALUE[sociodemographic_info][sociodemographic_group][language]}."
    
def query_template(query: str,
                   language: str = 'en') -> str:
    """
    Returns the query prompt block

    Args:
        query (str): The political issue to be discussed
        language (str): The language of the prompt to return

    Returns:
        str: The query prompt block
    """
    if language == 'en':
        return "You are asked about the following political issue: " + \
            f"{query}"
    elif language == 'de':
        return "Sie werden nach dem folgenden politischen Thema gefragt: " + \
            f"{query}"
    elif language == 'it':
        return "Le viene chiesto del seguente tema politico: " + \
            f"{query}"
    elif language == 'fr':
        return "Vous êtes interrogé sur le sujet politique suivant: " + \
            f"{query}"
    
def explanation_template(explanation: str,
                         language: str = 'en') -> str:
    """
    Returns the explanation prompt block
    
    Args:
        language (str): The language of the prompt to return

    Returns:
        str: The explanation prompt block
    """
    if language == 'en':
        return "\n\nThe following is an explanation of the political " + \
            f"issue:\n{explanation}"
    elif language == 'de':
        return "\n\nFolgendes ist eine Erklärung des politischen " + \
            f"Themas:\n{explanation}"
    elif language == 'it':
        return "\n\nLa seguente é una spiegazione del tema politico:\n" + \
            f"\n{explanation}"
    elif language == 'fr':
        return "\n\nVoici une explication du problème politique:\n" + \
            f"{explanation}"

def pros_cons_template(pros: str,
                       cons: str,
                       language: str = 'en') -> str:
    """
    Returns the pros and cons prompt block

    Args:
        pros (str): The pros of the political issue
        cons (str): The cons of the political issue
        language (str): The language of the prompt to return
    
    Returns:
        str: The pros and cons prompt block
    """
    if language == 'en':
        return "\n\nThe following are common arguments for the political issue:\n" + \
            f"{pros}\nThe following are common arguments against the political issue:\n" + \
            f"{cons}"
    elif language == 'de':
        return "\n\nFolgende sind gängige Argumente für das politische Thema:\n" + \
            f"{pros}\nFolgende sind gängige Argumente gegen das politische Thema:\n" + \
            f"{cons}"
    elif language == 'it':
        return "\n\nLe seguenti, sono argomentazioni tipiche a favore del tema politico:\n" + \
            f"{pros}\nLe seguenti, sono argomentazioni tipiche contrarie al tema politico:\n" + \
            f"{cons}"
    elif language == 'fr':
        return "\n\nVoici des arguments courants en faveur ce thème politique:\n" + \
            f"{pros}\nVoici des arguments courants contre le thème politique\n" + \
            f"{cons}"
    
def question_template(stance: str,
                      language: str = 'en') -> str:
    """
    Returns the question prompt block

    Args:
        stance (str): The stance of the politician
        language (str): The language of the prompt to return
    
    Returns:
        str: The question prompt block
    """    
    if language == 'en':
        return f"\n\nProvide a comment {STANCE[stance][language]} of the aforementioned political issue. You can use up to 500 characters."
        # return f"\n\nProvide an argument {STANCE[stance][language]} of the aforementioned political issue." # in favor / against
    elif language == 'de':
        return f"\n\nSchreiben Sie einen Kommentar {STANCE[stance][language]} die oben genannte politische Frage. Sie können bis zu 500 Zeichen verwenden."
        # return f"\n\nSchreiben Sie ein Argument {STANCE[stance][language]} die oben genannten politischen Frage."  # für / gegen
    elif language == 'it':
        return f"\n\nScriva un commento {STANCE[stance][language]} il tema politico precedentemente menzionato. Ha a disposizione fino a 500 caratteri."
        # return f"\n\nScriva un argomento {STANCE[stance][language]} il tema politico precedentemente menzionato." # a favore / contro 
    elif language == 'fr':
        return f"\n\nDonnez un commentaire {STANCE[stance][language]} thème politique mentionné précédemment. Vous pouvez utiliser jusqu'à 500 caractères."
        # return f"\n\nDonnez un argument {STANCE[stance][language]} thème politique mentionné précédemment."# en faveur du / contre le


def generate_prompt(query: str,
                    language: str,
                    context: bool,
                    sociodemographic_info: str|None = None,
                    sociodemographic_group: str|None = None,
                    explanation: str|None = None,
                    pros: str|None = None,
                    cons: str|None = None,
                    stance: str = 'in favor',
                    ) -> str:
    """
    Generates the prompt from a tuple of prompt blocks

    General structure:
        1. Role prompt block
            1.1 Context ('You are a Swiss politician...')
            1.2 Sociodemographic ('As a politician of the following sociodemographic group...')
        2. Context prompt block
            2.1 Query
            2.2 Explanation of the context of the given query
            2.3 Pros and Cons
        3. Question prompt block
            3.1 Instruction
            3.2 Stance

    Following this, a full example with all blocks would look like this:
        You are a Swiss politician filling out a questionnaire about your political beliefs for a voting advice application in the context of a federal election.
        You belong to the following sociodemographic group: Age - 18-24.

        You are asked about the following political issue:
        [QUERY]
        [EXPLANATION]
        [PROS and CONS]

        Provide an argument {in favor|against} of the aforementioned political issue.

    Args:
        query (str): The political issue to be discussed
        context (bool): Whether the context block should be included
        sociodemographic_info (str):
            The sociodemographic information of the politician
        sociodemographic_group (str):
            The sociodemographic group of the politician
        explanation (str): The explanation of the political issue
        pros (str): The pros of the political issue
        cons (str): The cons of the political issue
        stance (str): The stance of the politician
    """
    prompt = ""

    if context:
        prompt += context_template(language=language)
    if sociodemographic_info is not None and \
        sociodemographic_group is not None:
        prompt += sociodemographic_template(
            sociodemographic_info=sociodemographic_info,
            sociodemographic_group=sociodemographic_group,
            language=language)
    if len(prompt) > 0:  # Add a newline if context block was added
        prompt += "\n\n"
    
    prompt += query_template(query=query, language=language)

    if explanation is not None:
        prompt += explanation_template(explanation=explanation,
                                       language=language)
    if pros is not None and cons is not None:
        prompt += pros_cons_template(pros=pros,
                                     cons=cons,
                                     language=language)

    prompt += question_template(stance=stance, language=language)

    return prompt

