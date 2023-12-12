import chess.svg
from stockfish import Stockfish


class ChessGame:

    def __init__(self):
        # mapping of piece class names to FEN symbols
        self.piece_mapping = {
            'White-king': 'K',
            'White-queen': 'Q',
            'White-rook': 'R',
            'White-bishop': 'B',
            'White-knight': 'N',
            'White-pawn': 'P',
            'Black-king': 'k',
            'Black-queen': 'q',
            'Black-rook': 'r',
            'Black-bishop': 'b',
            'Black-knight': 'n',
            'Black-pawn': 'p',
        }
        self.depth = 20
        self.skill_level = 20
        self.stockfish = Stockfish("stockfish-windows-x86-64.exe")
        self.stockfish.set_depth(self.depth)  # TODO: look up what that means
        self.stockfish.set_skill_level(self.skill_level)  # 0 = lowest, 20 = highest
        self.current_board = None

    def set_skill_level(self, skill_level):
        self.skill_level = skill_level
        self.stockfish.set_skill_level(skill_level)
        print('Skill level set to ' + str(skill_level))

    def set_depth(self, depth):
        self.depth = depth
        self.stockfish.set_depth(depth)

    def _make_fen_string(self, recognized_pieces):
        # Initialize an empty 8x8 chessboard grid with spaces
        chessboard = [[' ' for _ in range(8)] for _ in range(8)]

        for piece in recognized_pieces:
            square = piece['square']
            class_name = piece['class_name']
            row = 8 - int(square[1])
            col = ord(square[0]) - ord('a')
            chessboard[row][col] = self.piece_mapping[class_name]

        # Generate the FEN string
        fen = '/'.join([''.join(row) for row in chessboard])
        fen = fen.replace(' ' * 8, '8')  # Replace consecutive empty squares with their count
        fen = fen.replace(' ' * 7, '7')
        fen = fen.replace(' ' * 6, '6')
        fen = fen.replace(' ' * 5, '5')
        fen = fen.replace(' ' * 4, '4')
        fen = fen.replace(' ' * 3, '3')
        fen = fen.replace(' ' * 2, '2')
        fen = fen.replace(' ' * 1, '1')

        return fen

    def get_fen_string(self, recognized_pieces):
        fen = self._make_fen_string(recognized_pieces)
        self.current_board = chess.Board(fen)
        self.current_board.turn = chess.WHITE
        valid = self.current_board.is_valid()
        return fen, valid

    """ 
    returns fen string, boolean for if the move was valid, boolean if black is check, 
    boolean if black is checkmate, boolean if white is check, boolean if white is checkmate, computer move as string
    """
    def make_move(self, recognized_pieces):
        fen = self._make_fen_string(recognized_pieces)
        new_board = chess.Board(fen)

        if new_board.is_check():
            return fen, True, False, False, False, True, None

        if not new_board.is_valid():
            print(new_board.status())
            print("Invalid board")
            return fen, False, False, False, False, False, None

        # make the players move
        for move in self.current_board.legal_moves:
            self.current_board.push(move)
            if self.current_board.board_fen() == new_board.board_fen():
                # now the computer needs make a move
                self.stockfish.set_fen_position(fen)
                self.current_board.turn = chess.BLACK
                self.current_board.push_san(self.stockfish.get_best_move())
                print("Computer move: " + str(self.current_board.peek()))
                # check if the white player is in check
                if self.current_board.is_check():
                    print("White is in check")
                    self.current_board.turn = chess.WHITE
                    return self.current_board.fen(), True, False, False, True, False, str(self.current_board.peek())
                elif self.current_board.is_checkmate():
                    print("White is in checkmate")
                    return fen, True, False, False, False, True, str(self.current_board.peek())
                else:  # no one is check
                    self.current_board.turn = chess.WHITE
                    return self.current_board.fen(), True, False, False, False, False, str(self.current_board.peek())
            _ = self.current_board.pop()

        else:
            print("Invalid move")
            return fen, False, False, False, False, False, None
