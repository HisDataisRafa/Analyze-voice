import librosa
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class VoiceToneAnalyzer:
    def __init__(self, file_path):
        """
        Inicializa el analizador con la ruta del archivo MP3
        
        Args:
            file_path (str): Ruta al archivo MP3 a analizar
        """
        self.file_path = file_path
        self.y = None
        self.sr = None
        self.pitch_values = None
        self.speaker_segments = None
        
    def load_audio(self):
        """Carga el archivo de audio usando librosa"""
        print("Cargando archivo de audio...")
        self.y, self.sr = librosa.load(self.file_path)
        print(f"Audio cargado: duración = {len(self.y)/self.sr:.2f} segundos")
        
    def extract_pitch(self):
        """Extrae los valores de pitch (F0) del audio"""
        print("Extrayendo características de pitch...")
        
        # Extraer el pitch usando el algoritmo PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        
        # Filtrar valores válidos de pitch
        self.pitch_values = f0[~np.isnan(f0)]
        return self.pitch_values
    
    def identify_speakers(self, n_speakers=2):
        """
        Identifica diferentes hablantes basándose en clusters de pitch
        
        Args:
            n_speakers (int): Número esperado de hablantes
        """
        print("Identificando hablantes...")
        
        # Usar KMeans para clustering de pitch
        kmeans = KMeans(n_clusters=n_speakers, random_state=42)
        pitch_2d = self.pitch_values.reshape(-1, 1)
        speaker_labels = kmeans.fit_predict(pitch_2d)
        
        # Organizar los clusters por frecuencia media (menor = masculino)
        centroids = kmeans.cluster_centers_.flatten()
        speaker_order = np.argsort(centroids)
        
        self.speaker_segments = {
            'Masculino': self.pitch_values[speaker_labels == speaker_order[0]],
            'Femenino': self.pitch_values[speaker_labels == speaker_order[1]]
        }
        
        return self.speaker_segments
    
    def analyze_tone_characteristics(self):
        """Analiza las características de tono para cada hablante"""
        tone_analysis = {}
        
        for speaker, pitches in self.speaker_segments.items():
            # Calcular estadísticas básicas
            stats = {
                'pitch_medio': np.mean(pitches),
                'pitch_std': np.std(pitches),
                'pitch_min': np.min(pitches),
                'pitch_max': np.max(pitches),
                'rango_tonal': np.max(pitches) - np.min(pitches)
            }
            
            # Clasificar el rango vocal
            if speaker == 'Masculino':
                if stats['pitch_medio'] < 110:
                    stats['tipo_voz'] = 'Bajo'
                elif stats['pitch_medio'] < 130:
                    stats['tipo_voz'] = 'Barítono'
                else:
                    stats['tipo_voz'] = 'Tenor'
            else:
                if stats['pitch_medio'] < 220:
                    stats['tipo_voz'] = 'Contralto'
                elif stats['pitch_medio'] < 260:
                    stats['tipo_voz'] = 'Mezzosoprano'
                else:
                    stats['tipo_voz'] = 'Soprano'
                    
            tone_analysis[speaker] = stats
            
        return tone_analysis
    
    def plot_pitch_distribution(self):
        """Genera un gráfico de la distribución de pitch para cada hablante"""
        plt.figure(figsize=(12, 6))
        
        for speaker, pitches in self.speaker_segments.items():
            density = gaussian_kde(pitches)
            xs = np.linspace(min(pitches), max(pitches), 200)
            plt.plot(xs, density(xs), label=speaker)
        
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Densidad')
        plt.title('Distribución de Pitch por Hablante')
        plt.legend()
        plt.grid(True)
        plt.savefig('pitch_distribution.png')
        plt.close()

def main():
    # Ejemplo de uso
    analyzer = VoiceToneAnalyzer('ruta_a_tu_archivo.mp3')
    analyzer.load_audio()
    analyzer.extract_pitch()
    analyzer.identify_speakers()
    
    # Analizar características de tono
    tone_analysis = analyzer.analyze_tone_characteristics()
    
    # Imprimir resultados
    for speaker, analysis in tone_analysis.items():
        print(f"\nAnálisis para {speaker}:")
        print(f"Tipo de voz: {analysis['tipo_voz']}")
        print(f"Pitch medio: {analysis['pitch_medio']:.2f} Hz")
        print(f"Rango tonal: {analysis['rango_tonal']:.2f} Hz")
        
    # Generar gráfico
    analyzer.plot_pitch_distribution()

if __name__ == "__main__":
    main()
