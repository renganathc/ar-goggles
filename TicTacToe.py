import cv2
import numpy as np
import math
import mediapipe as mp
import copy
import time

def CreateUI(BoardState,UIHeight,CellSize,X,O):
	UIBG = np.zeros((UIHeight,UIHeight,3),dtype=np.uint8)

	for i in range(1, 3):
		cv2.line(UIBG, (i * CellSize, 0), (i * CellSize, UIHeight), (255,255,255), 10)
		cv2.line(UIBG, (0, i * CellSize), (UIHeight, i * CellSize), (255,255,255), 10)
	for r in range(3):
		for c in range(3):
			center_x, center_y = c * CellSize + CellSize // 2, r * CellSize + CellSize // 2
			if BoardState[r][c] == X:
				cv2.line(UIBG, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (255, 0, 0), 6)
				cv2.line(UIBG, (center_x + 50, center_y - 50), (center_x - 50, center_y + 50), (255, 0, 0), 6)
			elif BoardState[r][c] == O:
				cv2.circle(UIBG, (center_x, center_y), 50, (0, 0, 255), 6)

	return UIBG

def CalculateUICorners(WFrame,HFrame,UIHeight):
    Factor = 2
    UITopLeftX,UITopLeftY = (WFrame-UIHeight)//Factor,(HFrame-UIHeight)//Factor
    UITopLeft = (UITopLeftX,UITopLeftY)
    UITopRight = (UITopLeftX+UIHeight,UITopLeftY)
    UIBottomLeft = (UITopLeftX,UITopLeftY+UIHeight)
    UIBottomRight = (UITopLeftX+UIHeight,UITopLeftY+UIHeight)
       
    return np.array([UITopLeft,UITopRight,UIBottomRight,UIBottomLeft], dtype=np.float32)

def Touch(frame,Matrix,WFrame,HFrame,UIBG,BoxDims,hand,mpdrawing,mphands):
    IsPinching = False
    FingerOption = -1
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    MpResult = hand.process(RGB)
    IndexPos = (0,0)
    
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
            
    return UIBG,FingerOption,IsPinching,IndexPos

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
    _,frame = cam.read()
    HFrame, WFrame, _ = frame.shape

    UIHeight = WFrame//2
    CellSize = UIHeight//3
    h,w = UIHeight,UIHeight
    board = InitialState(EMPTY)

    BoxDims = []
    for r in range(3):
        for c in range(3):
            BoxDims.append((c * CellSize, r * CellSize, CellSize, CellSize))
            
    LastPinch = 0
    PinchCooldown = 1.0

    BotThinking = False
    BotThinkingStart = 0
    BotDelay = 3

    Alpha = 0.8

    mphands = mp.solutions.hands
    mpdrawing = mp.solutions.drawing_utils
    hand = mphands.Hands(min_detection_confidence = 0.7,max_num_hands = 1)

    while True:
        success, frame = cam.read()
        if not success: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        HFrame, WFrame, _ = frame.shape
        FrameArea = HFrame*WFrame
        Matrix = None
        key = cv2.waitKey(1) & 0xFF

        
        DestinationPoints = CalculateUICorners(WFrame,HFrame,UIHeight)
        DestinationPointsInt = np.int32(DestinationPoints)

        TopLeftCorner = DestinationPointsInt[0]
        XStart = TopLeftCorner[0]
        YStart = TopLeftCorner[1]

        BoardWidth = DestinationPointsInt[1][0] - DestinationPointsInt[0][0]
        BoardHeight = DestinationPointsInt[3][1] - DestinationPointsInt[0][1]

        XEnd = XStart + BoardWidth
        YEnd = YStart + BoardHeight
        SourcePoints = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        Matrix, _ = cv2.findHomography(SourcePoints, DestinationPoints)
            
        UIBG = CreateUI(board,UIHeight,CellSize,X,O)
        UIBG, FingerOption, IsPinching, IndexPos = Touch(frame, Matrix, WFrame, HFrame, UIBG, BoxDims, hand, mpdrawing, mphands)
        GameOver = terminal(board,EMPTY)
        CurrentPlayer = Player(board,X,O)
        CurrentTime = time.time()
        
        if not GameOver and CurrentPlayer == X and not BotThinking:
            if FingerOption != -1 and IsPinching and (CurrentTime > LastPinch+PinchCooldown):
                row, col = FingerOption // 3, FingerOption % 3
                if board[row][col] == EMPTY:
                    board = result(board, (row, col),EMPTY,X,O)
                    LastPinch = CurrentTime

                    if not terminal(board,EMPTY) and Player(board,X,O) == O:
                        print("Schedule Bot Turn")
                        BotThinking = True
                        BotThinkingStart = CurrentTime
        if GameOver and IsPinching and (CurrentTime > LastPinch+PinchCooldown):
            break
        # PinchCooldown = IsPinching
        if BotThinking and (CurrentTime > BotThinkingStart+BotDelay):
            print("Bot's turn")
            move = minimax(board,X,EMPTY,O)
            board = result(board, move,EMPTY,X,O)
            BotThinking = False
        if Matrix is not None:
            roi = frame[YStart:YEnd,XStart:XEnd]
            UIBG = CreateUI(board,UIHeight,CellSize,X,O)
            if roi.shape[:2] == UIBG.shape[:2]:
                BlendedROI = cv2.addWeighted(roi,1-Alpha,UIBG,Alpha,0)
                frame[YStart:YEnd,XStart:XEnd] = BlendedROI
                cv2.circle(frame,IndexPos,25,(0,0,255),-1)

        message = ""
        if terminal(board,EMPTY):
            win = winner(board,EMPTY)
            if win is None: message = "Game Over: Tie. Pinch to quit!"
            else: message = f"Game Over: {win} wins. Pinch to quit!"
        elif BotThinking:
            message = "Bot is thinking..."
        else:
            message = f"Your Turn ({X})"
        (TextWidth, TextHeight), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 5)
        TextX = (WFrame-TextWidth+50)//2
        TextY = TextHeight

        frame = cv2.flip(frame,1)
        cv2.putText(frame, message, (TextX,TextY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        
        cv2.imshow("VideoFeed", frame)
        if  key == ord('q'):
            break

    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    TicTacToeMain()