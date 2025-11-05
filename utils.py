import glob
import torch

def get_part_model(part, fdir='./exp'):
    """
    지정된 디렉토리에서 특정 'part' 문자열을 포함하는
    가장 마지막(아마도 최신) 모델(.pt) 파일 경로를 찾습니다.

    Args:
        part (str): 파일 경로에 포함되어야 하는 문자열 (예: 'encoder').
        fdir (str, optional): 검색을 시작할 기본 디렉토리. Defaults to './exp'.

    Returns:
        str: 찾은 모델 파일의 경로.
        None: 파일을 찾지 못한 경우.
    """
    exps = []
    # fdir/*/*.pt 패턴으로 파일 검색 (하위 디렉토리까지 검색)
    exp_paths = glob.glob(f'{fdir}/*/*.pt', recursive=True)
    
    for exp in sorted(exp_paths):
        if part in exp:
            exps.append(exp)
    
    if not exps:
        print(f"경고: '{part}'에 해당하는 모델을 {fdir}에서 찾을 수 없습니다.")
        return None
    
    # 정렬된 리스트의 마지막 항목 (가장 최신 파일) 반환
    return exps[-1]

def load_model_part(part, fdir='./exp', device='cpu'):
    """
    get_part_model을 사용해 모델 경로를 찾고,
    해당 모델(.pt)을 불러와 device로 이동시킵니다.
    
    참고: .pt 파일이 모델의 state_dict가 아닌 모델 객체 자체를
    저장했다고 가정합니다.

    Args:
        part (str): 불러올 모델 부분 (예: 'encoder').
        fdir (str, optional): 검색할 디렉토리. Defaults to './exp'.
        device (str, optional): 모델을 로드할 장치 ('cpu' 또는 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.nn.Module: 로드된 모델 객체.
        None: 파일을 찾지 못하거나 로드에 실패한 경우.
    """
    model_path = get_part_model(part, fdir)
    
    if model_path is None:
        return None
        
    try:
        # map_location을 사용하여 지정된 장치로 바로 로드
        model = torch.load(model_path, map_location=device)
        
        # .pt 파일이 state_dict인 경우 (아래 주석 참고)
        # model_instance = YourModelClass(*args, **kwargs) 
        # model_instance.load_state_dict(torch.load(model_path, map_location=device))
        # model = model_instance
        
        model.to(device)
        model.eval() # 추론 모드로 설정
        print(f"성공: '{part}' 모델을 {model_path}에서 로드했습니다. (Device: {device})")
        return model
    except Exception as e:
        print(f"오류: {model_path}에서 모델 로드 실패 - {e}")
        return None

def feat_ext(data, model):
    """
    주어진 모델을 사용하여 데이터에서 피처(feature)를 추출합니다.
    model.forward()를 feat_ext=True 플래그와 함께 호출합니다.

    Args:
        data (torch.Tensor): 모델에 입력할 데이터 텐서.
        model (torch.nn.Module): 피처 추출에 사용할 모델 객체.

    Returns:
        torch.Tensor: 모델이 반환한 피처.
    """
    # 피처 추출(추론) 시에는 그래디언트 계산을 비활성화하는 것이 좋습니다.
    with torch.no_grad():
        feats = model.forward(data, feat_ext=True)
    return feats
