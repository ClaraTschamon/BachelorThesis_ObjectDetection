import chess.svg
from stockfish import Stockfish


class ChessGame:
    def __init__(self):
        # mapping of piece class names to FEN symbols
        self.__piece_mapping = {
            'white-king': 'K',
            'white-queen': 'Q',
            'white-rook': 'R',
            'white-bishop': 'B',
            'white-knight': 'N',
            'white-pawn': 'P',
            'black-king': 'k',
            'black-queen': 'q',
            'black-rook': 'r',
            'black-bishop': 'b',
            'black-knight': 'n',
            'black-pawn': 'p',
        }
        self.__depth = 20
        self.__skill_level = 20
        self.__stockfish = Stockfish("stockfish-windows-x86-64.exe")
        self.__stockfish.set_depth(self.__depth)  # TODO: look up what that means
        self.__stockfish.set_skill_level(self.__skill_level)  # 0 = lowest, 20 = highest

    def get_skill_level(self):
        return self.__skill_level

    def set_skill_level(self, skill_level):  # TODO: implement in UI
        self.__skill_level = skill_level
        self.__stockfish.set_skill_level(skill_level)

    def get_depth(self):
        return self.__depth

    def set_depth(self, depth):
        self.__depth = depth
        self.__stockfish.set_depth(depth)

    def __make_fen_string(self, recognized_pieces):
        # Initialize an empty 8x8 chessboard grid with spaces
        chessboard = [[' ' for _ in range(8)] for _ in range(8)]

        for piece in recognized_pieces:
            square = piece['square']
            class_name = piece['class_name']
            row = 8 - int(square[1])
            col = ord(square[0]) - ord('a')
            chessboard[row][col] = self.__piece_mapping[class_name]

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

    def get_fen(self, recognized_pieces):
        fen = self.__make_fen_string(recognized_pieces)
        return fen

    def make_next_move(self, recognized_pieces):
        board = self.get_fen(recognized_pieces)
        fen = board.fen()
        self.__stockfish.set_fen_position(fen)
        board.turn = chess.BLACK
        board.push_san(self.__stockfish.get_best_move())
        return board.fen()

    def get_is_check(self, fen):
        board = chess.Board(fen)
        return board.is_check()

    def get_is_checkmate(self, fen):
        board = chess.Board(fen)
        return board.is_checkmate()
