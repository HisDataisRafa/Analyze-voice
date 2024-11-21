import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# Importar librosa de manera segura
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.error("""
        Error: No se pudo cargar la librería librosa. 
        Por favor, asegúrate de que todas las dependencias están instaladas correctamente.
        Ejecuta: pip install -r requirements.txt
    """)

class SimpleVoiceAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.y = None
        self.sr = None
        
    def load_audio(self):
        """Carga el archivo de audio de manera segura"""
        try:
            self.y, self.sr = librosa.load(self.file_path)
            return True
        except Exception as e:
            st.error(f"Error al cargar el audio: {str(e)}")
            return False
            
    def analyze_voice(self):
        """Análisis básico de voz"""
        try:
            # Extraer características
            pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr)
            
            # Calcular estadísticas básicas
            pitch_mean = np.mean(pitches[pitches > 0])
            pitch_std = np.std(pitches[pitches > 0])
            
            # Clasificación simple
            if pitch_mean < 150:  # Umbral aproximado entre voz masculina y femenina
                voice_type = "Masculina"
            else:
                voice_type = "Femenina"
                
            return {
                'tipo_voz': voice_type,
                'pitch_medio': pitch_mean,
                'pitch_std': pitch_std
            }
            
        except Exception as e:
            st.error(f"Error en el análisis: {str(e)}")
            return None

def main():
    st.title("🎤 Analizador de Voz Simplificado")
    
    st.write("""
    ### Sube un archivo de audio para analizar las características de la voz
    Formatos soportados: MP3, WAV
    """)
    
    if not LIBROSA_AVAILABLE:
        st.warning("La aplicación no puede funcionar sin librosa. Por favor, instala todas las dependencias.")
        return
        
    uploaded_file = st.file_uploader("Elige un archivo de audio", type=['mp3', 'wav'])
    
    if uploaded_file is not None:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filename = tmp_file.name
            
        try:
            with st.spinner('Analizando el audio...'):
                analyzer = SimpleVoiceAnalyzer(temp_filename)
                
                if analyzer.load_audio():
                    # Realizar análisis
                    results = analyzer.analyze_voice()
                    
                    if results:
                        st.success("¡Análisis completado!")
                        
                        # Mostrar resultados
                        st.subheader("Resultados del Análisis")
                        st.write(f"**Tipo de Voz Detectada:** {results['tipo_voz']}")
                        st.write(f"**Pitch Medio:** {results['pitch_medio']:.2f} Hz")
                        st.write(f"**Desviación Estándar del Pitch:** {results['pitch_std']:.2f} Hz")
                        
                        # Crear visualización simple
                        fig, ax = plt.subplots()
                        librosa.display.waveshow(analyzer.y, sr=analyzer.sr, ax=ax)
                        ax.set_title('Forma de Onda del Audio')
                        st.pyplot(fig)
                        
        except Exception as e:
            st.error(f"Error inesperado: {str(e)}")
            
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    st.markdown("""
    ### Instrucciones
    1. Sube un archivo de audio (MP3 o WAV)
    2. Espera mientras se procesa el archivo
    3. Revisa los resultados del análisis
    
    ### Notas
    - Para mejores resultados, usa grabaciones claras y con poco ruido
    - El archivo no debe exceder los 50MB
    - Se recomienda una duración entre 5 y 30 segundos
    """)

if __name__ == "__main__":
    main()
    main()
