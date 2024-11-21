import streamlit as st
import librosa
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import tempfile

class VoiceToneAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.y = None
        self.sr = None
        self.pitch_values = None
        self.speaker_segments = None
        
    def load_audio(self):
        """Carga el archivo de audio usando librosa"""
        self.y, self.sr = librosa.load(self.file_path)
        return len(self.y)/self.sr
        
    def extract_pitch(self):
        """Extrae los valores de pitch (F0) del audio"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr
        )
        
        self.pitch_values = f0[~np.isnan(f0)]
        return self.pitch_values
    
    def identify_speakers(self, n_speakers=2):
        """Identifica diferentes hablantes basándose en clusters de pitch"""
        if len(self.pitch_values) == 0:
            return None
            
        kmeans = KMeans(n_clusters=n_speakers, random_state=42)
        pitch_2d = self.pitch_values.reshape(-1, 1)
        speaker_labels = kmeans.fit_predict(pitch_2d)
        
        centroids = kmeans.cluster_centers_.flatten()
        speaker_order = np.argsort(centroids)
        
        self.speaker_segments = {
            'Masculino': self.pitch_values[speaker_labels == speaker_order[0]],
            'Femenino': self.pitch_values[speaker_labels == speaker_order[1]]
        }
        
        return self.speaker_segments
    
    def analyze_tone_characteristics(self):
        """Analiza las características de tono para cada hablante"""
        if self.speaker_segments is None:
            return None
            
        tone_analysis = {}
        
        for speaker, pitches in self.speaker_segments.items():
            stats = {
                'pitch_medio': np.mean(pitches),
                'pitch_std': np.std(pitches),
                'pitch_min': np.min(pitches),
                'pitch_max': np.max(pitches),
                'rango_tonal': np.max(pitches) - np.min(pitches)
            }
            
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
        if self.speaker_segments is None:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for speaker, pitches in self.speaker_segments.items():
            density = gaussian_kde(pitches)
            xs = np.linspace(min(pitches), max(pitches), 200)
            ax.plot(xs, density(xs), label=speaker)
        
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('Densidad')
        ax.set_title('Distribución de Pitch por Hablante')
        ax.legend()
        ax.grid(True)
        
        return fig

def main():
    st.title("Analizador de Tonos de Voz")
    st.write("""
    Esta aplicación analiza archivos de audio para identificar y clasificar diferentes voces,
    distinguiendo entre voces masculinas y femeninas.
    """)
    
    uploaded_file = st.file_uploader("Sube un archivo de audio (MP3)", type=['mp3'])
    
    if uploaded_file is not None:
        # Crear un archivo temporal para guardar el audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
        
        try:
            with st.spinner('Analizando el audio...'):
                # Crear instancia del analizador
                analyzer = VoiceToneAnalyzer(temp_filename)
                
                # Cargar y analizar el audio
                duration = analyzer.load_audio()
                st.info(f"Duración del audio: {duration:.2f} segundos")
                
                # Extraer pitch y identificar hablantes
                analyzer.extract_pitch()
                analyzer.identify_speakers()
                
                # Analizar características de tono
                tone_analysis = analyzer.analyze_tone_characteristics()
                
                if tone_analysis:
                    # Mostrar resultados
                    st.subheader("Resultados del Análisis")
                    
                    for speaker, analysis in tone_analysis.items():
                        st.write(f"\n**{speaker}**:")
                        st.write(f"- Tipo de voz: {analysis['tipo_voz']}")
                        st.write(f"- Pitch medio: {analysis['pitch_medio']:.2f} Hz")
                        st.write(f"- Rango tonal: {analysis['rango_tonal']:.2f} Hz")
                    
                    # Mostrar gráfico
                    st.subheader("Distribución de Pitch")
                    fig = analyzer.plot_pitch_distribution()
                    if fig:
                        st.pyplot(fig)
                else:
                    st.error("No se pudieron detectar voces claramente en el audio.")
                    
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
            
        finally:
            # Limpiar el archivo temporal
            os.unlink(temp_filename)
    
    st.markdown("""
    ### Instrucciones de uso:
    1. Sube un archivo MP3 que contenga voces de dos personas (preferiblemente un hombre y una mujer)
    2. Espera mientras el sistema analiza el audio
    3. Revisa los resultados del análisis y la visualización
    
    ### Notas:
    - El audio debe tener buena calidad para mejores resultados
    - Se recomienda que los hablantes hablen por separado
    - La duración óptima del audio es entre 10 y 60 segundos
    """)

if __name__ == "__main__":
    main()
