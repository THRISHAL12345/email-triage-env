def grade(response: str, expected_output: str) -> dict:
    try:
        score = 1.0 if response.strip() == expected_output.strip() else 0.0
        return {
            "score": score,
            "feedback": "Correct" if score == 1.0 else "Incorrect"
        }
    except Exception as e:
        return {
            "score": 0.0,
            "feedback": f"Error: {str(e)}"
        }