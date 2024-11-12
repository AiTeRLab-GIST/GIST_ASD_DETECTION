import pdb
import os
import PySimpleGUI as sg
import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from asd_model.model import Wav2Vec2ForSpeechClassification as Model
from transformers import AutoConfig, Wav2Vec2Processor
from scipy.special import softmax
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
""" Microphone waveform display with Pyaudio, Pyplot and PysimpleGUI """

class GUI:
    def __init__(self):
        sg.theme('Default1')
        AppFont = 'Arial'
        self.timeout = 128

        # 오디오 데이터 스트리밍을 위한 버퍼 사전 정의
        self.audioData = np.array([])

        # wavefile 포인터 사전 정의
        self.waveFile = None
        self.pAud = pyaudio.PyAudio()

        # 오디오 frame 관련 내용 정의
        self.fridx = 0
        self.frame_rate = 16000
        self.frame_size = 2048 # 0.128 second
        
        # 오디오 데이터 ploting을 위한 x, y축 정의 --> frame_rate*5, 5초 구간 표시
        self.xData = np.linspace(0, self.frame_rate*5, num=self.frame_rate*5, dtype=int)
        self.yData = np.zeros(self.frame_rate*5)
        
        # 영유아 자폐 진단 확률 ploting을 위한 x, y축 정의 --> frame_rate*5, 5초 구간 표시
        self.xProb = np.linspace(0, self.frame_rate*5 // self.frame_size, num=self.frame_rate*5 // self.frame_size, dtype=int)
        self.yProb = np.zeros(self.frame_rate*5//self.frame_size)
        
        # softmax output probability 평균 값 (최종 판별 결과) 계산을 위한 empty list 사전 정의
        self.probs_stacks = [] 
        
        # samples 폴더 내의 파일 목록 조회
        # self.sample_list = os.listdir('samples')
        
        # 중단 버튼 클릭 시 정지를 위한 stop flag
        self.stop_flag = False

        """중요한 부분"""
        # transformers 모듈로 pre-trained wav2vec2.0 config, model, dataprocessor load
        exp_name = './asd_model/exp/checkpoint'
        self.config = AutoConfig.from_pretrained(exp_name)
        self.model = Model.from_pretrained(exp_name, config=self.config).to('cpu')
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        """       """
        # wave plot 캔버스
        streaming_col = [
            [sg.Canvas(key='figCanvas')],
            [sg.Text('', font=('Arial',4))],
            [sg.ProgressBar(4000, orientation='h',size=(65, 30), key='progress_bar')],
            [sg.Text('', font=('Arial',4))],
            [sg.Button('실행', font=('Arial',18)),
             sg.Button('중단', font=('Arial',18), disabled = True),
             sg.Button('종료', font=('Arial',18))]]
            
        # asd 확률 캔버스
        decision_col = [
            [sg.Text("음성 샘플 폴더", font=('Arial',18)),
             sg.In(size=(10,1), font=('Arial',18), enable_events=True, key='folder'),
             sg.FolderBrowse(target='folder', initial_folder='.')],
            [sg.Text('')],
            [sg.Listbox(values=[], enable_events=True, auto_size_text=True, size=(50,20), key='flist')],
            [sg.Text('')],
            [sg.Text('판별 결과', font=('Arial',18)), sg.In(size=(12,12), font=('Arial',18), enable_events=True, key='decision')],
            [sg.Text('', font=('Arial',6))],
            [sg.Text('판별 확률', font=('Arial',18)), sg.In(size=(12,12), font=('Arial',18), enable_events=True, key='prob')]]

        # 전체 layout
        layout = [[sg.Column(streaming_col), sg.VSeperator(),
                   sg.Column(decision_col)]]

        self.window = sg.Window('ASD Detection DEMO', layout, finalize = True, size=(1100,625), location = (400, 100))

    """중요한 부분"""
    def inference(self, waveform):
        """
        input wave (1초 길이)에 대한 ASD 확률 계산을 위한 함수
        input : waveform
        output : probability
        """
        features = self.processor(waveform, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=False)
        input_values = features.input_values.to('cpu')
        logits = self.model(input_values).logits.cpu().detach()

        return softmax(logits)
    """        """
    
    def start_stream(self):
        """
        실시간 waveform streaming을 위한 함수
        """
        self.window['중단'].update(disabled = False)
        self.window['실행'].update(disabled = True)
        self.stream = self.pAud.open(format=pyaudio.paInt16,
                      channels=self.waveFile.getnchannels(),
                      rate=self.waveFile.getframerate(),
                      output=True,
                      frames_per_buffer=self.frame_size,
                      stream_callback=self.callback)
        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status):
        """
        1. buff에 1 frame의 오디오를 불러옴
        2. self.yData의 1~5 frame초 구간을 0~4 frame 구간으로 이동
        3. buff의 오디오를 self.yData의 4~5 frame 구간에 입력
        4. self.yProb 또한 2, 3과 동일한 과정으로 probability 업데이트
        5. self.yData의 4~5 frame 데이터를 inference 하여 probability 추정
        업데이트 된 
        
        """
        buff = np.zeros((self.frame_size,))
        probs = np.zeros((self.frame_rate*5 // self.frame_size,))
        self.audioData = np.frombuffer(self.waveFile.readframes(self.frame_size), dtype=np.int16)
        buff[:len(self.audioData)] = self.audioData
        self.yData[:-self.frame_size] = self.yData[self.frame_size:]
        self.yData[-self.frame_size:] = buff
        self.yProb[:-1] = self.yProb[1:]
        prob = self.inference(self.yData[-self.frame_rate:])[0]
        self.yProb[-1] = prob
        self.probs_stacks.append(prob)
        self.decision = 'ASD' if self.yProb.mean() > 0.5 else 'TD'
        self.mean_prob = np.mean(self.probs_stacks)
        return (self.audioData, pyaudio.paContinue)

    def stop(self, event):
        """
        stop 함수 호출 시 중단 및 실행 버튼 모양 바꾸고,
        종료 버튼 눌러졌을 때에는 오디오 스트리밍 종료
        """
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.window['progress_bar'].update(0)
            self.window['중단'].update(disabled=True)
            self.window['실행'].update(disabled=False)
        if event == '종료':
            self.pAud.terminate()

    def draw_figure(self, canvas, figure):
        """
        캔버스 및 figure 입력을 통해 전체 widget을 설정하는 함수
        """
        fig_agg = FigureCanvasTkAgg(figure, canvas)
        fig_agg.draw()
        fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return fig_agg

    def plot_canvas(self):
        """
        Canvas 내에 2개의 figure를 drawing 하기 위한 함수 (matplotlib 모듈)
        waveform figure 및 probability figure를 그림
        """
        # waveform figure plotting
        self.fig = plt.figure()
        self.fig_ax = self.fig.add_subplot(211) #subplot으로 상단에 waveform plot
        self.fig_ax.set_title('Children Speech Signal')
        self.fig_ax.set_xlabel('time')
        self.fig_ax.set_ylim(-32768, 32767)
        self.fig_ax.autoscale(enable=True, axis='x', tight=True)
        self.fig_ax.set_xticks([])
        self.fig_ax.set_yticks([0])
        self.fig_line, = self.fig_ax.plot(self.xData, self.yData)
        
        # probability figure plotting
        self.asd_ax = self.fig.add_subplot(212) #subplot으로 상단에 ASD probability plot
        self.asd_ax.set_title('ASD Probability')
        self.asd_ax.set_xlabel('time')
        self.asd_ax.set_ylim(0, 1)
        self.asd_ax.autoscale(enable=True, axis='x', tight=True)
        self.asd_ax.set_xticks([])
        #self.asd_ax.set_yticks([])
        self.asd_line, = self.asd_ax.plot(self.xProb, self.yProb, 'k--')
        self.fig.tight_layout(pad = 1.0)
        self.fig_agg = self.draw_figure(self.window['figCanvas'].TKCanvas, self.fig)
    
    def update_canvas(self):
        """
        오디오 1 frame 읽을 때마다 캔버스 업데이트
        yData에 waveform data, yProb에 probability data가
        매 frame 갱신됨 (callback 함수 참조)
        """
        self.fridx += 1
        self.fig_line.set_ydata(self.yData)
        self.asd_line.set_ydata(self.yProb)
        self.fig_agg.draw()
        self.fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    def clear_canvas(self):
        """ 스트리밍 종료 시 캔버스 내용을 지움 """
        self.fridx = 0
        self.yData = np.zeros(self.yData.shape)
        self.yProb = np.zeros(self.yProb.shape)
        self.fig_line.set_ydata(self.yData.shape)
        self.asd_line.set_ydata(self.yProb.shape)
        self.fig_agg.draw()
        self.fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    def get_audioFile(self):
        fname = os.path.join(self.folder, self.fname)
        self.waveFile = wave.open(fname)
        self.n_frames = self.waveFile.getnframes() // self.frame_size
        self.channels = self.waveFile.getnchannels()
        self.frame_rate = self.waveFile.getframerate()
    
    def process(self):
        plt.style.use('ggplot')
        self.plot_canvas()
        
        while True:
            event, values = self.window.read(timeout=self.timeout)
                
            if event == 'folder':
                self.folder = values[event]
                flist = os.listdir(self.folder)
                self.window['flist'].update(flist)
                    
            if event == 'flist':
                self.fname = values[event][0]

            if event == sg.WIN_CLOSED or event == '종료':
                self.stop(event)
                break

            if event == '실행':
                self.get_audioFile()
                self.start_stream()
                self.stop_flag = False
            
            elif event == '중단':
                self.stop(event)
                self.clear_canvas()
                self.stop_flag = True
                self.probs_stacks = []
                self.window['decision'].update('')
                self.window['prob'].update('')
                
            
            elif self.audioData.size != 0 and self.stop_flag is not True:
                self.window['progress_bar'].update(self.fridx * (4000 / (self.n_frames-5)))
                if self.fridx == int(self.n_frames):
                    self.window['decision'].update(self.decision)
                    if self.decision == 'ASD':
                        self.window['prob'].update(self.mean_prob)
                    else:
                        self.window['prob'].update(1-self.mean_prob)
                self.update_canvas()

if __name__ == '__main__':
    gui = GUI()
    gui.process()