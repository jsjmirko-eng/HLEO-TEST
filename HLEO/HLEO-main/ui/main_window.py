from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QHBoxLayout,
)
from controllers.search_controller import SearchController
from PySide6.QtWidgets import QTableWidgetItem

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.controller = SearchController()
        self.search_button.clicked.connect(self.start_search)

        self.setWindowTitle("HLEO v1.0")
        self.resize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        title = QLabel("HLEO - Hair Loss Evidence Organizer")
        layout.addWidget(title)

        search_layout = QHBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Inserisci una ricerca...")

        self.search_button = QPushButton("Cerca")

        search_layout.addWidget(self.search_box)
        search_layout.addWidget(self.search_button)

        layout.addLayout(search_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Fonte",
            "Titolo",
            "Anno",
            "Score",
        ])
        
        layout.addWidget(self.table)
        
        def start_search(self):
    query = self.search_box.text()

    results = self.controller.search(query)

    self.table.setRowCount(len(results))

    for row, result in enumerate(results):

        source = result["type"].capitalize()

        title = ""
        year = ""
        score = "-"

        if result["type"] == "pubmed":
            article = result["article"]
            title = article.title
            year = str(article.year or "")

        elif result["type"] == "europepmc":
            article = result["article"]
            title = article.title
            year = str(article.year or "")

        elif result["type"] == "clinicaltrials":
            trial = result["trial"]
            title = trial.title
            year = str(trial.year or "")

        elif result["type"] == "reddit":
            post = result["post"]
            title = post.title
            year = ""

        self.table.setItem(row, 0, QTableWidgetItem(source))
        self.table.setItem(row, 1, QTableWidgetItem(title))
        self.table.setItem(row, 2, QTableWidgetItem(year))
        self.table.setItem(row, 3, QTableWidgetItem(score))