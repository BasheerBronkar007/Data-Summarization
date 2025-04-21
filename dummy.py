import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}  

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)


    print("Status Code:", response.status_code)
    print("Raw Response:", response.text)

    if response.status_code != 200:
        print("Error:", response.text)
        return None

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        return None


input_text = """India, officially the Republic of India,[j][21] is a country in South Asia. It is the seventh-largest country by area; the most populous country from June 2023 onwards;[22][23] and since its independence in 1947, the world's most populous democracy.[24][25][26] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar, and Indonesia.

Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[28][29][30] Their long occupation, predominantly in isolation as hunter-gatherers, has made the region highly diverse, second only to Africa in human genetic diversity.[31] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[32] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest.[33][34] Its hymns recorded the dawning of Hinduism in India.[35] India's pre-existing Dravidian languages were supplanted in the northern regions.[36] By 400 BCE, caste had emerged within Hinduism,[37] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[38] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires.[39] Widespread creativity suffused this era,[40] but the status of women declined,[41] and untouchability became an organized belief.[l][42] In South India, the Middle kingdoms exported Dravidian language scripts and religious cultures to the kingdoms of Southeast Asia.[43]"""

params = {
    "inputs": input_text,
    "parameters": {
        "max_length": 50,  # Maximum length of the summary
        "min_length": 20,  # Minimum length of the summary
        "do_sample": True,  # Enable sampling (for more diverse summaries)
        "temperature": 0.7,  # Lower values = more deterministic, higher = more random
        "top_k": 50,  # Limits sampling to top 50 words (reduces randomness)
        "top_p": 0.9,  # Nucleus sampling: keeps words with top 90% probability
        "no_repeat_ngram_size": 3  # Prevents repeating n-grams (phrases)
    }
}

'''
output = query({
    "inputs": "India, officially the Republic of India,[j][21] is a country in South Asia. It is the seventh-largest country by area; the most populous country from June 2023 onwards;[22][23] and since its independence in 1947, the world's most populous democracy.[24][25][26] Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west;[k] China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives; its Andaman and Nicobar Islands share a maritime border with Thailand, Myanmar, and Indonesia.

Modern humans arrived on the Indian subcontinent from Africa no later than 55,000 years ago.[28][29][30] Their long occupation, predominantly in isolation as hunter-gatherers, has made the region highly diverse, second only to Africa in human genetic diversity.[31] Settled life emerged on the subcontinent in the western margins of the Indus river basin 9,000 years ago, evolving gradually into the Indus Valley Civilisation of the third millennium BCE.[32] By 1200 BCE, an archaic form of Sanskrit, an Indo-European language, had diffused into India from the northwest.[33][34] Its hymns recorded the dawning of Hinduism in India.[35] India's pre-existing Dravidian languages were supplanted in the northern regions.[36] By 400 BCE, caste had emerged within Hinduism,[37] and Buddhism and Jainism had arisen, proclaiming social orders unlinked to heredity.[38] Early political consolidations gave rise to the loose-knit Maurya and Gupta Empires.[39] Widespread creativity suffused this era,[40] but the status of women declined,[41] and untouchability became an organized belief.[l][42] In South India, the Middle kingdoms exported Dravidian language scripts and religious cultures to the kingdoms of Southeast Asia.[43]",
})

print("Final Output:", output)
'''
output = query(params)


if output and isinstance(output, list) and "summary_text" in output[0]:
    print("\n **Generated Summary:**", output[0]["summary_text"])
else:
    print("\n Failed to generate summary.")
