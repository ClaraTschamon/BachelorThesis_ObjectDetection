import chess.svg
from stockfish import Stockfish


class ChessGame:
    def __init__(self):
        # mapping of piece class names to FEN symbols
        self.__piece_mapping = {
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
        self.__depth = 20
        self.__skill_level = 20
        self.__stockfish = Stockfish("stockfish-windows-x86-64.exe")
        self.__stockfish.set_depth(self.__depth)  # TODO: look up what that means
        self.__stockfish.set_skill_level(self.__skill_level)  # 0 = lowest, 20 = highest
        self.current_board = None

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

    def get_fen_string(self, recognized_pieces):
        fen = self.__make_fen_string(recognized_pieces)

        # if it is not the first capture of the board:
        # check if the move made is valid if it was valid return the fen
        #if self.current_board is not None:
        #    new_board = chess.Board(fen)
        #    for move in self.current_board.legal_moves:
        #        self.current_board.push(move)
        #        if self.current_board.board_fen() == new_board.board_fen():
        #            return fen, True  # valid move
        #        _ = self.current_board.pop()
        #    else:
        #        return fen, False  # invalid fen
        #else:
        #    self.current_board = chess.Board(fen)
        #    return fen, True  # function was called for the first time
        self.current_board = chess.Board(fen)
        return fen

    def make_move(self, recognized_pieces):
        fen = self.__make_fen_string(recognized_pieces)
        print('Make computer move called, FEN: ' + fen)

        # check if players move was valid and then make the computer move one black figure
        new_board = chess.Board(fen)
        print(self.current_board)
        print(self.current_board.legal_moves)
        for move in self.current_board.legal_moves:
            self.current_board.push(move)
            if self.current_board.board_fen() == new_board.board_fen():
                # it was a valid move and now the computer needs make a move

                try:
                    self.__stockfish.set_fen_position(fen)
                    self.current_board.turn = chess.BLACK
                    self.current_board.push_san(self.__stockfish.get_best_move())
                    self.current_board.turn = chess.WHITE
                    return self.current_board.fen(), True
                except:
                    return fen, False

            _ = self.current_board.pop()
        else:
            return fen, False  # players move was invalid


    def get_is_check(self, fen):
        board = chess.Board(fen)
        return board.is_check()

    def get_is_checkmate(self, fen):
        board = chess.Board(fen)
        return board.is_checkmate()
