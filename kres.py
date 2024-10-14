import tkinter as tk
from tkinter import messagebox


class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Крестики-нолики")
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.current_player = "X"
        self.create_buttons()
        self.reset_button = tk.Button(
            self.root, text="Перезапуск", command=self.reset_game
        )
        self.reset_button.grid(row=3, column=0, columnspan=3)

    def create_buttons(self):
        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    self.root,
                    text="",
                    width=10,
                    height=3,
                    command=lambda i=i, j=j: self.on_button_click(i, j),
                )
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

    def on_button_click(self, i, j):
        if self.buttons[i][j]["text"] == "" and self.check_winner() is False:
            self.buttons[i][j]["text"] = self.current_player
            if self.check_winner():
                messagebox.showinfo("Победа!", f"Игрок {self.current_player} выиграл!")
            elif self.is_draw():
                messagebox.showinfo("Ничья", "Игра завершилась вничью!")
            else:
                self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        for i in range(3):
            if (
                self.buttons[i][0]["text"]
                == self.buttons[i][1]["text"]
                == self.buttons[i][2]["text"]
                != ""
            ):
                return True
        for j in range(3):
            if (
                self.buttons[0][j]["text"]
                == self.buttons[1][j]["text"]
                == self.buttons[2][j]["text"]
                != ""
            ):
                return True
        if (
            self.buttons[0][0]["text"]
            == self.buttons[1][1]["text"]
            == self.buttons[2][2]["text"]
            != ""
        ):
            return True
        if (
            self.buttons[0][2]["text"]
            == self.buttons[1][1]["text"]
            == self.buttons[2][0]["text"]
            != ""
        ):
            return True
        return False

    def is_draw(self):
        for i in range(3):
            for j in range(3):
                if self.buttons[i][j]["text"] == "":
                    return False
        return True

    def reset_game(self):
        self.current_player = "X"
        for i in range(3):
            for j in range(3):
                self.buttons[i][j]["text"] = ""


if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
