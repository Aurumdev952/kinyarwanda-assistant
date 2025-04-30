from qa_data import QA_PAIRS


def get_answer_basic(question):
    question = question.strip().lower()
    response = None
    for k, v in QA_PAIRS.items():
        if k in question:
            response = v
            break
    if response is None:
        response = "Mbabarira, nta gisubizo mfite kuri icyo kibazo."
    return response
