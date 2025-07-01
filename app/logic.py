from app.load import df, model, index
import numpy as np

def get_diagnosis(symptom_text, user_age, user_gender):
    query_embedding = model.encode([symptom_text])
    D, I = index.search(np.array(query_embedding), k=5)

    results = []
    for score, idx in zip(D[0], I[0]):
        row = df.iloc[idx]
        age_penalty = abs(row['Age'] - user_age)
        gender_match = row['Gender'].lower() == user_gender.lower()
        adjusted_score = score + (0.1 * age_penalty) - (0.2 if gender_match else 0)
        results.append((adjusted_score, row))

    # Deduplicate by disease
    best_by_disease = {}
    for score, row in results:
        disease = row['Disease']
        if disease not in best_by_disease or score < best_by_disease[disease][0]:
            best_by_disease[disease] = (score, row)
    deduped_results = sorted(best_by_disease.values(), key=lambda x: x[0])

    return format_whatsapp_response(deduped_results)

def format_whatsapp_response(results, top_n=3):
    messages = []
    for score, row in results[:top_n]:
        msg = (
            f"🔍 *Possible Disease:* {row['Disease']}\n"
            f"👨‍⚕️ *Specialist:* {row['Specialist']}\n"
            f"🩺 *Symptoms:* {', '.join(row['Symptoms_List'])}\n"
            f"📊 *Patient Profile:* Age {row['Age']}, {row['Gender']}\n"
            f"💓 *BP / Cholesterol:* {row['Blood Pressure']} / {row['Cholesterol Level']}\n"
            f"📉 *Match Score:* {score:.2f}"
        )
        messages.append(msg)
    return "\n\n".join(messages)
