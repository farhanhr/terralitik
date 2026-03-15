import google.generativeai as genai

def get_ai_recommendation(location, current_risk, avg_forecast, est_loss, api_key):
    if not api_key:
        return "API Key tidak ditemukan. Silakan tambahkan GEMINI_API_KEY di Streamlit Secrets."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
        Anda adalah asisten AI 'Terralitik', pakar mitigasi bencana kekeringan pertanian di Indonesia.
        Buatkan ringkasan eksekutif dan instruksi mitigasi singkat (maksimal 3 paragraf) untuk wilayah {location}.
        
        Data saat ini:
        - Status Risiko Saat Ini: {current_risk}
        - Rata-rata Skor Kekeringan 30 Hari Kedepan: {avg_forecast * 100:.1f}% (Batas krisis > 75%)
        - Estimasi Kerugian Finansial: Rp {est_loss:,.0f} per Hektar
        
        Format output:
        1. [Analisis Singkat] - Jelaskan makna data di atas dengan bahasa yang mudah dipahami petani/aparat desa.
        2. [Tindakan Mitigasi] - Berikan 3 poin tindakan (bullet points) praktis dan spesifik untuk menyelamatkan panen.
        Gunakan bahasa Indonesia yang profesional, tegas, dan darurat jika berisiko tinggi.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Terjadi kesalahan pada sistem NLP: {str(e)}"