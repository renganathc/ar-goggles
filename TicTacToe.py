import apriltag
import cv2
import numpy as np
import math
import mediapipe as mp
import copy

def CreateUI(BoardState,UIHeight,CellSize,X,O):
	UIBG = np.zeros((UIHeight,UIHeight,3),dtype=np.uint8)

	for i in range(1, 3):
		cv2.line(UIBG, (i * CellSize, 0), (i * CellSize, UIHeight), (1, 1, 1), 40)
		cv2.line(UIBG, (0, i * CellSize), (UIHeight, i * CellSize), (1, 1, 1), 40)
	for r in range(3):
		for c in range(3):
			center_x, center_y = c * CellSize + CellSize // 2, r * CellSize + CellSize // 2
			if BoardState[r][c] == X:
				cv2.line(UIBG, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (0, 0, 255), 6)
				cv2.line(UIBG, (center_x + 50, center_y - 50), (center_x - 50, center_y + 50), (0, 0, 255), 6)
			elif BoardState[r][c] == O:
				cv2.circle(UIBG, (center_x, center_y), 50, (255, 0, 0), 6)

	return UIBG

def CalculateUICorners(Tag):
	MarkerTopLeft,MarkerTopRight,MarkerBottomLeft = Tag.corners[0],Tag.corners[1],Tag.corners[3]
	MarkerCenter = Tag.center
	TopVector = MarkerTopRight - MarkerTopLeft
	LeftVector = MarkerBottomLeft - MarkerTopLeft
	OffsetScale = 3
	UIScale = 4
	UICenter = MarkerCenter+(LeftVector*OffsetScale)
	UIWidth = TopVector*UIScale
	UIHeight = LeftVector*UIScale
	UITopLeft = UICenter-(UIWidth/2)-(UIHeight/2)
	UITopRight = UITopLeft+UIWidth
	UIBottomLeft = UITopLeft+UIHeight
	UIBottomRight = UITopLeft+UIWidth+UIHeight

	return np.array([UITopLeft,UITopRight,UIBottomRight,UIBottomLeft], dtype=np.float32)

def Touch(frame,Matrix,WFrame,HFrame,UIBG,BoxDims,hand,mpdrawing,mphands):
    IsPinching = False
    FingerOption = -1
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    MpResult = hand.process(RGB)
    
    if MpResult.multi_hand_landmarks and Matrix is not None:
        HandLandmarks = MpResult.multi_hand_landmarks[0]
        mpdrawing.draw_landmarks(frame, HandLandmarks, mphands.HAND_CONNECTIONS)
        IndexTip = HandLandmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
        ThumbTip = HandLandmarks.landmark[mphands.HandLandmark.THUMB_TIP]
        IndexPos = (int(IndexTip.x * WFrame),int(IndexTip.y * HFrame))
        try:
            InvMatrix = np.linalg.inv(Matrix)
            FingerPosOnUI = cv2.perspectiveTransform(np.array([[IndexPos]], dtype=np.float32), InvMatrix)
            fx,fy = int(FingerPosOnUI[0][0][0]),int(FingerPosOnUI[0][0][1])
            for i,(bx,by,bw,bh) in enumerate(BoxDims):
                if bx<fx<bx+bw and by<fy<by+bh:
                    FingerOption = i
                    break
        except np.linalg.LinAlgError:
            print("LinAlgError!")
        dist = math.hypot(ThumbTip.x - IndexTip.x , ThumbTip.y-IndexTip.y)
        IsPinching = dist < 0.05
        
        if FingerOption != -1 and IsPinching:
            print(f"User selected box {FingerOption+1}")
            (bx,by,bw,bh) = BoxDims[FingerOption]
            cv2.rectangle(UIBG, (bx,by), (bx+bw,by+bh), (0,255,0), -1)
            
    return UIBG,FingerOption,IsPinching

def FindMarker(FrameArea,ApriltagResults):
	for r in ApriltagResults:
		MarkerArea = cv2.contourArea(np.array(r.corners, dtype=np.int32))
		if r.tag_id == 0 and MarkerArea > (FrameArea*0.008):
			return r
	return None		

def InitialState(EMPTY):
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]

def Player(board,X,O):
    XCount = sum(row.count(X) for row in board)
    OCount = sum(row.count(O) for row in board)
    return X if XCount <= OCount else O

def actions(board,EMPTY):
    PossibleActions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                PossibleActions.add((i, j))
    return PossibleActions

def result(board, action, EMPTY,X,O):
    if board[action[0]][action[1]] is not EMPTY:
        raise ValueError("Invalid action")
    NewBoard = copy.deepcopy(board)
    NewBoard[action[0]][action[1]] = Player(board,X,O)
    return NewBoard

def winner(board,EMPTY):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] is not EMPTY: return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] is not EMPTY: return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] is not EMPTY: return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] is not EMPTY: return board[0][2]
    return None

def utility(board,X,O,EMPTY):
    if winner(board,EMPTY) == X:
          return 1
    elif winner(board,EMPTY) == O:
        return -1
    else:
          return 0

def terminal(board,EMPTY):
    if winner(board,EMPTY) is not None or not any(EMPTY in row for row in board):
        return True
    return False

def minimax(board,X,EMPTY,O):
    if terminal(board,EMPTY):
        return None

    current_player = Player(board,X,O)

    if current_player == X:
        best_value = -float("inf")
        best_action = None
        for action in actions(board):
            value = min_value(result(board, action, EMPTY,X,O),EMPTY,X,O)
            if value > best_value:
                best_value = value
                best_action = action

    else:
        best_value = float("inf")
        best_action = None
        for action in actions(board,EMPTY):
            value = max_value(result(board, action, EMPTY,X,O),EMPTY,X,O)
            if value < best_value:
                best_value = value
                best_action = action
    return best_action

def max_value(board,EMPTY,X,O):
    if terminal(board,EMPTY):
        return utility(board,X,O,EMPTY)

    v = -float("inf")
    for action in actions(board,EMPTY):
        v = max(v, min_value(result(board, action, EMPTY,X,O),EMPTY,X,O))
    return v

def min_value(board,EMPTY,X,O):
    if terminal(board,EMPTY):
        return utility(board,X,O,EMPTY)
    v = float("inf")
    for action in actions(board,EMPTY):
        v = min(v, max_value(result(board, action, EMPTY,X,O),EMPTY,X,O))
    return v

def TicTacToeMain(cam):
    X = "X"
    O = "O"
    EMPTY = None
    UIHeight = 500
    CellSize = UIHeight//3
    h,w = UIHeight,UIHeight
    board = InitialState(EMPTY)

    BoxDims = []
    for r in range(3):
        for c in range(3):
            BoxDims.append((c * CellSize, r * CellSize, CellSize, CellSize))
            
    PinchCooldown = False
    Alpha = 0.6

    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(options)
    mphands = mp.solutions.hands
    mpdrawing = mp.solutions.drawing_utils
    hand = mphands.Hands(min_detection_confidence = 0.7,max_num_hands = 1)

    KalmanFilters = [cv2.KalmanFilter(4,2) for _ in range(4)]
    for kf in KalmanFilters:
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2

    FramesSinceDetection = 0

    while True:
        success, frame = cam.read()
        if not success: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        HFrame, WFrame, _ = frame.shape
        FrameArea = HFrame*WFrame
        ApriltagResults = detector.detect(gray)
        Marker = FindMarker(FrameArea,ApriltagResults)
        Matrix = None
        key = cv2.waitKey(1) & 0xFF

        PredictedCorners = np.array([kf.predict()[:2].flatten() for kf in KalmanFilters], dtype=np.float32)
        
        if Marker is not None:
            DestinationPoints = CalculateUICorners(Marker)
            FramesSinceDetection = 0
            for i,corner in enumerate(DestinationPoints):
                KalmanFilters[i].correct(corner)
        else:
            FramesSinceDetection += 1
        
        SmoothPoints = np.array([kf.statePost[:2].flatten() for kf in KalmanFilters], dtype=np.float32)

        if FramesSinceDetection <= 3:
            SourcePoints = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            Matrix, _ = cv2.findHomography(SourcePoints, SmoothPoints)
            
            UIBG = CreateUI(board,UIHeight,CellSize,X,O)
            UIBG, FingerOption, IsPinching = Touch(frame, Matrix, WFrame, HFrame, UIBG, BoxDims, hand, mpdrawing, mphands)
            GameOver = terminal(board,EMPTY)
            CurrentPlayer = Player(board,X,O)
            
            if not GameOver and CurrentPlayer == X:
                if FingerOption != -1 and IsPinching and not PinchCooldown:
                    row, col = FingerOption // 3, FingerOption % 3
                    if board[row][col] == EMPTY:
                        board = result(board, (row, col),EMPTY,X,O)
            if GameOver and IsPinching and not PinchCooldown:
                break
            PinchCooldown = IsPinching
            if not terminal(board,EMPTY) and Player(board,X,O) == O:
                print("Bot's turn")
                move = minimax(board,X,EMPTY,O)
                board = result(board, move,EMPTY,X,O)
            if Matrix is not None:
                WarpedUI = cv2.warpPerspective(UIBG, Matrix, (WFrame, HFrame))
                mask = np.sum(WarpedUI, axis=2) > 0
                if np.any(mask):
                    frame[mask] = cv2.addWeighted(frame[mask], 1-Alpha, WarpedUI[mask], Alpha, 0)

            message = ""
            if terminal(board,EMPTY):
                win = winner(board,EMPTY)
                if win is None: message = "Game Over: Tie. Pinch to quit!."
                else: message = f"Game Over: {win} wins. Pinch to quit!."
            else:
                message = f"Your Turn ({X})"
            cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        cv2.imshow("VideoFeed", frame)
        if  key == ord('q'):
            break

    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    TicTacToeMain()