import os
import webbrowser
import threading
import time
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__) #inizializza l'app Flask

class StudyRecommendationSystem: #sistema di raccomandazione metodi di studio
    def __init__(self): 
        self.model = None #modello di machine learning
        self.encoders = {} #dizionario per gli encoder delle variabili categoriche
        self.study_methods = [ #metodi di studio disponibili
            'Leitura ativa e anotações',
            'Flashcards e repetição espaçada',
            'Mapas mentais e conceituais',
            'Resolução prática de problemas',
            'Estudo em grupo',
            'Vídeos e materiais multimídia',
            'Resumos e esquemas',
            'Simulações e exercícios'
        ]
        self.resources = { #risorse associate a ciascun metodo di studio
            'Leitura ativa e anotações': ['Anki para revisão', 'Notion para organização', 'Modelo Cornell Notes'],
            'Flashcards e repetição espaçada': ['Anki', 'Quizlet', 'RemNote'],
            'Mapas mentais e conceituais': ['MindMeister', 'XMind', 'SimpleMind'],
            'Resolução prática de problemas': ['Khan Academy', 'Exercícios da Coursera', 'Plataformas de programação'],
            'Estudo em grupo': ['Grupos de estudo no Discord', 'Sessões no Zoom', 'Fóruns de estudo'],
            'Vídeos e materiais multimídia': ['YouTube educacional', 'TED-Ed', 'Vídeos da Coursera'],
            'Resumos e esquemas': ['Obsidian', 'Modelos do Notion', 'Ferramentas de mapas mentais'],
            'Simulações e exercícios': ['Testes práticos', 'Software de simulação', 'Plataformas de quiz']
        }
        
    def generate_training_data(self): #genera dati di training simulati per il modello
        np.random.seed(42) #42 è il seme per la riproducibilità
    
        learning_styles = ['Visual', 'Auditivo', 'Cinestésico', 'Leitura/Escrita'] #stili di apprendimento
        subjects = ['Matemática', 'Ciências', 'História', 'Línguas', 'Literatura', 'Informática', 'Arte', 'Filosofia']  #materie
        time_available = ['1-2 horas', '3-4 horas', '5+ horas'] #tempo disponibile (ore)
        difficulty = ['Baixa', 'Média', 'Alta'] #difficoltà percepita
        data = [] #lista per memorizzare i dati generati

        for _ in range(1000):  #genera 1000 esempi
            style = np.random.choice(learning_styles) #stile di apprendimento
            subject = np.random.choice(subjects) #materia
            time = np.random.choice(time_available) #tempo disponibile
            diff = np.random.choice(difficulty) #difficoltà percepita
            
            #logica per assegnare metodi di studio
            if style == 'Visual': #se lo stile è visuale
                if subject in ['Matemática', 'Ciências']: #se la materia è matematica o scienze
                    method = np.random.choice(['Mapas mentais e conceituais', 'Simulações e exercícios']) #sceglie tra questi metodi
                else: #altrimenti
                    method = np.random.choice(['Mapas mentais e conceituais', 'Vídeos e materiais multimídia']) #sceglie tra questi metodi
            elif style == 'Auditivo':
                if diff == 'Alta':
                    method = np.random.choice(['Estudo em grupo', 'Vídeos e materiais multimídia'])
                else:
                    method = np.random.choice(['Leitura ativa e anotações', 'Estudo em grupo'])
            elif style == 'Cinestésico':
                if subject in ['Matemática', 'Ciências', 'Informática']:
                    method = np.random.choice(['Resolução prática de problemas', 'Simulações e exercícios'])
                else:
                    method = np.random.choice(['Flashcards e repetição espaçada', 'Estudo em grupo'])
            else: 
                if time == '1-2 horas':
                    method = np.random.choice(['Resumos e esquemas', 'Flashcards e repetição espaçada'])
                else:
                    method = np.random.choice(['Leitura ativa e anotações', 'Resumos e esquemas'])
            
            data.append([style, subject, time, diff, method]) #aggiunge l'esempio ai dati
        
        return pd.DataFrame(data, columns=['learning_style', 'subject', 'time_available', 'difficulty', 'study_method']) #ritorna il DataFrame con i dati generati
    
    def train_model(self): #addestra il modello di machine learning
        df = self.generate_training_data() #genera dati di training
        
        features = ['learning_style', 'subject', 'time_available', 'difficulty'] #caratteristiche di input
        target = 'study_method' #variabile target
        
        X = df[features].copy() #dati di input
        y = df[target] #variabile target
        
        for feature in features: #per ogni caratteristica
            le = LabelEncoder() #crea un encoder
            X[feature] = le.fit_transform(X[feature]) #codifica la caratteristica
            self.encoders[feature] = le #salva l'encoder nel dizionario
        
        target_encoder = LabelEncoder() #encoder per la variabile target
        y_encoded = target_encoder.fit_transform(y) #codifica la variabile target
        self.encoders['target'] = target_encoder #salva l'encoder nel dizionario
        
        #suddivide i dati in training e test (20% per il test, 80% per il training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        #addestra il modello
        self.model = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators è il numero di alberi nella foresta e random_state è per la riproducibilità
        self.model.fit(X_train, y_train) #addestra il modello sui dati di training
        
        accuracy = self.model.score(X_test, y_test) #calcola l'accuratezza sui dati di test
        print(f"Accuratezza del modello: {accuracy:.2f}") #stampa l'accuratezza del modello
    
    def predict(self, learning_style, subject, time_available, difficulty):
        #a una predizione basata sui parametri di input
        if not self.model: #se il modello non è addestrato
            self.train_model() #addestra il modello
        
        input_data = pd.DataFrame([[learning_style, subject, time_available, difficulty]], #crea un DataFrame con i dati di input
                                columns=['learning_style', 'subject', 'time_available', 'difficulty'])
        
        for feature in input_data.columns: #per ogni caratteristica
            if feature in self.encoders: #se esiste un encoder per la caratteristica
                try:
                    input_data[feature] = self.encoders[feature].transform(input_data[feature]) #codifica la caratteristica
                except ValueError:
                    input_data[feature] = 0 #se il valore non è stato visto durante il training, usa il primo valore
        
        prediction = self.model.predict(input_data)[0] #fa la predizione
        predicted_method = self.encoders['target'].inverse_transform([prediction])[0] #decodifica la predizione
        
        probabilities = self.model.predict_proba(input_data)[0] #ottiene le probabilità per ogni classe
        method_probabilities = {}
        for i, prob in enumerate(probabilities): #per ogni probabilità
            method = self.encoders['target'].inverse_transform([i])[0] #decodifica la classe
            method_probabilities[method] = prob #salva la probabilità nel dizionario
        
        return predicted_method, method_probabilities #ritorna il metodo predetto e le probabilità associate
    
    def generate_study_plan(self, method, time_available, difficulty):
        #genera il piano di studio personalizzato
        time_map = {'1-2 horas': 90, '3-4 horas': 210, '5+ horas': 300} #mappa del tempo disponibile in minuti
        total_minutes = time_map.get(time_available, 120) #ottiene il tempo totale disponibile in minuti, default 120
        
        if method == 'Leitura ativa e anotações': #se il metodo è lettura attiva e appunti
            sessions = [ #imposta le sessioni di studio
                f"Leitura preliminar (15 min)",
                f"Leitura aprofundada com anotações ({total_minutes//2} min)",
                f"Revisão e síntese ({total_minutes//4} min)",
                f"Revisão final ({total_minutes//4} min)"
            ]
        elif method == 'Flashcards e repetição espaçada':
            sessions = [
                f"Criação de flashcards ({total_minutes//3} min)",
                f"Primeira sessão de revisão ({total_minutes//3} min)",
                f"Revisão espaçada ({total_minutes//3} min)"
            ]
        elif method == 'Mapas mentais e conceituais':
            sessions = [
                f"Brainstorming de conceitos ({total_minutes//4} min)",
                f"Criação do mapa mental ({total_minutes//2} min)",
                f"Revisão e conexões ({total_minutes//4} min)"
            ]
        elif method == 'Resolução prática de problemas':
            sessions = [
                f"Estudo da teoria básica ({total_minutes//3} min)",
                f"Resolução de problemas guiados ({total_minutes//2} min)",
                f"Prática independente ({total_minutes//6} min)"
            ]
        elif method == 'Estudo em grupo':
            sessions = [
                f"Preparação individual ({total_minutes//4} min)",
                f"Discussão em grupo ({total_minutes//2} min)",
                f"Síntese coletiva ({total_minutes//4} min)"
            ]
        elif method == 'Vídeos e materiais multimídia':
            sessions = [
                f"Visualização de vídeo introdutório ({total_minutes//3} min)",
                f"Aprofundamento multimídia ({total_minutes//3} min)",
                f"Revisão ativa ({total_minutes//3} min)"
            ]
        elif method == 'Resumos e esquemas':
            sessions = [
                f"Leitura e sublinhado ({total_minutes//3} min)",
                f"Criação de resumo ({total_minutes//2} min)",
                f"Revisão do esquema ({total_minutes//6} min)"
            ]
        else: 
            sessions = [
                f"Estudo preparatório ({total_minutes//4} min)",
                f"Simulação prática ({total_minutes//2} min)",
                f"Análise de erros ({total_minutes//4} min)"
            ]
        return sessions

study_system = StudyRecommendationSystem() #inizializza il sistema di raccomandazione

#template HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="../books.png" type="image/x-icon">
    <title>Sistema AI - Raccomandazione Metodi di Studio</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            padding: 40px;
        }

        .form-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }

        .form-section h2 {
            color: #333;
            margin-bottom: 25px;
            font-size: 1.8em;
            font-weight: 600;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
            font-size: 1.1em;
        }

        .form-group select {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 1em;
            background: white;
            transition: all 0.3s ease;
        }

        .form-group select:focus {
            outline: none;
            border-color: #4facfe;
            box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
        }
        .form-group select:hover {
            cursor: pointer;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            cursor: pointer;
        }

        .results-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        }

        .results-section h2 {
            color: #333;
            margin-bottom: 25px;
            font-size: 1.8em;
            font-weight: 600;
        }

        .recommendation-card {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }

        .recommendation-card h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.4em;
        }

        .confidence-bar {
            background: #e1e5e9;
            height: 8px;
            border-radius: 4px;
            margin: 10px 0;
            overflow: hidden;
        }

        .confidence-fill {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            border-radius: 4px;
            transition: width 0.8s ease;
        }

        .study-plan {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 1px solid #e1e5e9;
        }

        .study-plan h4 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .session-item {
            background: #f8f9fa;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 3px solid #4facfe;
        }

        .resources-list {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border: 1px solid #e1e5e9;
        }

        .resources-list h4 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }

        .resource-item {
            background: #e8f4fd;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 6px;
            color: #2c5aa0;
            font-weight: 500;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .hidden {
            display: none;
        }

        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
                gap: 20px;
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .header p {
                font-size: 1em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Recomendador de Estudo com IA</h1>
            <p>Sistema inteligente para otimizar seus métodos de estudo</p>
        </div>

        <div class="content">
            <div class="form-section">
                <h2>📝 Seus Parâmetros de Estudo</h2>
                <form id="studyForm">
                    <div class="form-group">
                        <label for="learning_style">🧠 Estilo de Aprendizagem:</label>
                        <select id="learning_style" name="learning_style" required>
                            <option value="">Selecione seu estilo...</option>
                            <option value="Visual">Visual - Prefiro imagens e diagramas</option>
                            <option value="Auditivo">Auditivo - Aprendo melhor ouvindo</option>
                            <option value="Cinestésico">Cinestésico - Aprendo praticando</option>
                            <option value="Leitura/Escrita">Leitura/Escrita - Prefiro textos</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="subject">📚 Matéria de Estudo:</label>
                        <select id="subject" name="subject" required>
                            <option value="">Selecione a matéria...</option>
                            <option value="Matemática">Matemática</option>
                            <option value="Ciências">Ciências</option>
                            <option value="História">História</option>
                            <option value="Línguas">Línguas</option>
                            <option value="Literatura">Literatura</option>
                            <option value="Informática">Informática</option>
                            <option value="Arte">Arte</option>
                            <option value="Filosofia">Filosofia</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="time_available">⏰ Tempo Disponível:</label>
                        <select id="time_available" name="time_available" required>
                            <option value="">Selecione o tempo...</option>
                            <option value="1-2 horas">1-2 horas por dia</option>
                            <option value="3-4 horas">3-4 horas por dia</option>
                            <option value="5+ horas">5+ horas por dia</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="difficulty">📈 Dificuldade Percebida:</label>
                        <select id="difficulty" name="difficulty" required>
                            <option value="">Selecione a dificuldade...</option>
                            <option value="Baixa">Baixa - É bem fácil</option>
                            <option value="Média">Média - Exige esforço</option>
                            <option value="Alta">Alta - É bastante desafiador</option>
                        </select>
                    </div>

                    <button type="submit" class="btn">🔍 Obter Recomendação da IA</button>
                </form>
            </div>

            <div class="results-section">
                <h2>🎯 Recomendações Personalizadas</h2>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>A IA está analisando seus parâmetros...</p>
                </div>

                <div id="results" class="hidden">
                    <div class="recommendation-card">
                        <h3>🏆 Método Recomendado</h3>
                        <div id="recommended-method"></div>
                        <div class="confidence-bar">
                            <div id="confidence-fill" class="confidence-fill"></div>
                        </div>
                        <small id="confidence-text"></small>
                    </div>

                    <div class="study-plan">
                        <h4>📅 Plano de Estudo Personalizado</h4>
                        <div id="study-sessions"></div>
                    </div>

                    <div class="resources-list">
                        <h4>🛠️ Recursos Recomendados</h4>
                        <div id="recommended-resources"></div>
                    </div>
                </div>

                <div id="no-results" class="hidden">
                    <p style="text-align: center; color: #666; font-size: 1.1em; padding: 40px;">
                        Preencha o formulário para obter recomendações personalizadas! 🚀
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('studyForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Mostra loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.add('hidden');
            document.getElementById('no-results').classList.add('hidden');
            
            // Raccogli i dati del form
            const formData = new FormData(this);
            const data = {
                learning_style: formData.get('learning_style'),
                subject: formData.get('subject'),
                time_available: formData.get('time_available'),
                difficulty: formData.get('difficulty')
            };
            
            try {
                // Invia richiesta al server
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                // Nascondi loading
                document.getElementById('loading').style.display = 'none';
                
                if (result.success) {
                    // Mostra risultati
                    displayResults(result);
                    document.getElementById('results').classList.remove('hidden');
                } else {
                    alert('Errore nel ottenere raccomandazioni: ' + result.error);
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Errore di connessione: ' + error.message);
            }
        });
        
        function displayResults(result) {
            // Metodo raccomandato
            document.getElementById('recommended-method').innerHTML = 
                `<strong style="font-size: 1.2em; color: #667eea;">${result.recommended_method}</strong>`;
            
            // Barra di confidenza
            const confidence = Math.round(result.confidence * 100);
            document.getElementById('confidence-fill').style.width = confidence + '%';
            document.getElementById('confidence-text').textContent = 
                `Confidenza: ${confidence}%`;
            
            // Piano di studio
            const sessionsHtml = result.study_plan.map(session => 
                `<div class="session-item">${session}</div>`
            ).join('');
            document.getElementById('study-sessions').innerHTML = sessionsHtml;
            
            // Risorse consigliate
            const resourcesHtml = result.resources.map(resource => 
                `<div class="resource-item">${resource}</div>`
            ).join('');
            document.getElementById('recommended-resources').innerHTML = resourcesHtml;
        }
        
        // Mostra messaggio iniziale
        document.getElementById('no-results').classList.remove('hidden');
    </script>
</body>
</html>
"""

@app.route('/') #path principale
def index(): #renderizza la pagina principale
    return render_template_string(HTML_TEMPLATE) #

@app.route('/predict', methods=['POST']) #endpoint per le predizioni
def predict():
    try:
        data = request.json #ottieni i dati dalla richiesta
        
        #estrae i parametri
        learning_style = data['learning_style']
        subject = data['subject']
        time_available = data['time_available']
        difficulty = data['difficulty']
        
        #fa la predizione
        recommended_method, probabilities = study_system.predict(
            learning_style, subject, time_available, difficulty
        )
        
        #genera il piano di studio
        study_plan = study_system.generate_study_plan(
            recommended_method, time_available, difficulty
        )

        resources = study_system.resources.get(recommended_method, []) #ottiene le risorse associate al metodo raccomandato
        
        confidence = probabilities[recommended_method] #ottiene la confidenza della predizione
        
        return jsonify({ #ritorna i risultati come JSON
            'success': True,
            'recommended_method': recommended_method,
            'confidence': confidence,
            'study_plan': study_plan,
            'resources': resources,
            'all_probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def open_browser(): #apre il browser
    time.sleep(1.0)  #attende che il server sia avviato
    webbrowser.open('http://127.0.0.1:5000/') #apre l'URL nel browser predefinito

if __name__ == '__main__':
    print("Inizializzazione Sistema AI per Raccomandazione Metodi di Studio...")
    
    #addestra il modello
    print("Training del modello AI in corso...")
    study_system.train_model()
    print("Modello AI addestrato con successo!")
    
    #avvia il browser in un thread separato
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("Avvio server Flask...")
    print("Apertura applicazione nel browser")
    print("URL: http://127.0.0.1:5000/")
    
    #avvia l'applicazione Flask
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=5000)