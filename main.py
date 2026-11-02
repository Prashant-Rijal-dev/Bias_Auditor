"""Bias Auditor - Tkinter app. Run:  python main.py"""
import os
import tempfile
import tkinter as tk
import webbrowser
from tkinter import ttk, filedialog, messagebox, simpledialog

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import analysis as an
import charts
import experiments as ex
from models import GENERATORS


class BiasAuditorApp:
    def __init__(self, root):
        self.root = root
        root.title("Bias Auditor")
        root.geometry("1150x700")
        root.minsize(900, 560)
        self.df = None
        self.numeric = []
        self.cat_cols = ["gender", "region"]
        self.before = self.after = None
        self.canvas = None

        self.status = tk.Label(root, text="Ready. Click 'Generate data' (Ctrl+G) to begin.",
                               anchor="w", relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)                     # pack manager
        self._build_menu()
        self._build_controls()
        self._build_output()
        self._bind_keys()

    # ------------------------------------------------------------ layout
    def _build_menu(self):
        bar = tk.Menu(self.root)
        f = tk.Menu(bar, tearoff=0)
        f.add_command(label="Save dataset as CSV...", command=self.save_csv)
        f.add_command(label="Load CSV...", command=self.load_csv)
        f.add_command(label="Save report...", command=self.save_report)
        f.add_separator()
        f.add_command(label="Exit", command=self.root.destroy)
        bar.add_cascade(label="File", menu=f)

        e = tk.Menu(bar, tearoff=0)
        items = [("1a  Twin primes < 1000", self.e1a), ("1b  Permutations & combinations", self.e1b),
                 ("2a  List operations", self.e2a), ("2b  User lookup (dictionary)", self.e2b),
                 ("3   Student class", self.e3), ("4a  Swap NumPy columns", self.e4a),
                 ("4b  Pandas Sales filter", self.e4b), None,
                 ("5a  Bar graph", lambda: self.draw(ex.exp5_bar())),
                 ("5b  Pie chart", lambda: self.draw(ex.exp5_pie())),
                 ("5c  y = x\u00b2 plot", lambda: self.draw(ex.exp5_square())),
                 ("6   pack / grid / place window", lambda: ex.geometry_window(self.root))]
        for it in items:
            e.add_separator() if it is None else e.add_command(label=it[0], command=it[1])
        bar.add_cascade(label="Experiments", menu=e)

        h = tk.Menu(bar, tearoff=0)
        h.add_command(label="How to use", command=self.how_to)
        bar.add_cascade(label="Help", menu=h)
        self.root.config(menu=bar)

    def _build_controls(self):
        p = tk.Frame(self.root, padx=10, pady=10)
        p.pack(side=tk.LEFT, fill=tk.Y)
        p.columnconfigure(0, weight=1)
        p.columnconfigure(1, weight=1)

        def head(text, row):
            tk.Label(p, text=text, font=("Arial", 11, "bold")).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(10, 2))

        head("1. Dataset", 0)
        self.scenario = ttk.Combobox(p, values=list(GENERATORS), state="readonly")
        self.scenario.current(0)
        self.scenario.grid(row=1, column=0, columnspan=2, sticky="ew")
        tk.Label(p, text="Built-in bias strength (0 = fair)").grid(row=2, column=0, columnspan=2, sticky="w")
        self.bias = tk.Scale(p, from_=0, to=2, resolution=0.1, orient=tk.HORIZONTAL)
        self.bias.set(1.0)
        self.bias.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.randomize = tk.BooleanVar(value=False)
        tk.Checkbutton(p, text="New random data each time", variable=self.randomize).grid(
            row=4, column=0, columnspan=2, sticky="w")
        tk.Button(p, text="Generate data  (Ctrl+G)", command=self.generate).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=3)

        head("2. Audit", 6)
        tk.Label(p, text="Protected attribute").grid(row=7, column=0, sticky="w")
        self.group = ttk.Combobox(p, values=self.cat_cols, state="readonly", width=10)
        self.group.current(0)
        self.group.grid(row=7, column=1, sticky="e")
        self.use_sensitive = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text="Model can see gender/region", variable=self.use_sensitive).grid(
            row=8, column=0, columnspan=2, sticky="w")
        tk.Button(p, text="Run audit  (Ctrl+R)", command=self.audit).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=3)

        head("3. Fix", 10)
        tk.Button(p, text="Apply reweighing fix  (Ctrl+F)", command=self.fix).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=3)

        head("4. Charts", 12)
        self.chart_choice = ttk.Combobox(p, state="readonly", values=[
            "Approval bar chart", "Heatmap (two attributes)", "Violin (first numeric feature)",
            "Pie (share of approvals)", "Correlation matrix"])
        self.chart_choice.current(0)
        self.chart_choice.grid(row=13, column=0, columnspan=2, sticky="ew")
        tk.Button(p, text="Show chart", command=self.show_chart).grid(row=14, column=0, sticky="ew", pady=3, padx=(0, 2))
        tk.Button(p, text="Plotly 3D", command=self.show_plotly).grid(row=14, column=1, sticky="ew", pady=3, padx=(2, 0))

    def _build_output(self):
        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        box = tk.Frame(right)
        box.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.text = tk.Text(box, height=11, font=("Menlo" if self.root.tk.call("tk", "windowingsystem") == "aqua" else "Consolas", 10))
        sb = tk.Scrollbar(box, command=self.text.yview)
        self.text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chart_frame = tk.Frame(right)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        self.popup = tk.Menu(self.root, tearoff=0)                       # right-click menu
        self.popup.add_command(label="Clear report", command=lambda: self.text.delete("1.0", tk.END))
        self.popup.add_command(label="Copy all", command=self.copy_report)
        for ev in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):    # Windows/Linux and macOS
            self.text.bind(ev, lambda e: self.popup.tk_popup(e.x_root, e.y_root))

    def _bind_keys(self):
        for key, fn in (("r", self.audit), ("f", self.fix), ("g", self.generate)):
            self.root.bind(f"<Control-{key}>", lambda e, fn=fn: fn())
            self.root.bind(f"<Command-{key}>", lambda e, fn=fn: fn())    # macOS

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def say(self, msg):
        self.status.config(text=msg)

    def copy_report(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text.get("1.0", tk.END))

    def save_report(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt")
        if p:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self.text.get("1.0", tk.END))

    def need_data(self):
        if self.df is None:
            messagebox.showinfo("No data", "Generate or load a dataset first (Ctrl+G).")
            return False
        return True

    def draw(self, fig):
        for w in self.chart_frame.winfo_children():
            w.destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def report(self, title, res):
        self.log(f"--- {title} ---")
        for g, r in res["rates"].items():
            self.log(f"  {g:<10} approval rate: {r:.1%}")
        self.log(f"  Accuracy {res['accuracy']:.1%} | Disparate impact {res['di']:.2f} -> {an.verdict(res['di'])}")
        self.log(f"  Parity difference {res['spd']:.2f} | Equal-opportunity difference {res['eod']:.2f}")

    def run(self, mitigate):
        return an.run_audit(self.df, self.numeric, self.group.get(), sensitive_cols=self.cat_cols,
                            use_sensitive=self.use_sensitive.get(), mitigate=mitigate)[0]

    # ------------------------------------------------------------ main actions
    def generate(self):
        cls = GENERATORS[self.scenario.get()]
        gen = cls(n=3000, bias_strength=self.bias.get(), seed=None if self.randomize.get() else 42)
        self.df, filled = an.clean(gen.generate())
        self.numeric = gen.numeric_features
        self.cat_cols = list(gen.sensitive_columns)
        self.group.config(values=self.cat_cols)
        self.group.current(0)
        self.before = self.after = None
        self.log(f"[Data] {len(self.df)} rows | bias strength {self.bias.get()} | {filled} missing values filled with median")
        self.say(f"Dataset ready: {self.scenario.get()} ({len(self.df)} rows). Now click 'Run audit'.")

    def audit(self):
        if not self.need_data():
            return
        self.before, self.after = self.run(False), None
        self.report(f"AUDIT by {self.group.get()}", self.before)
        self.draw(charts.selection_bar(self.before["rates"], None, self.group.get()))
        self.say("Audit done. Click 'Apply reweighing fix' to compare.")

    def fix(self):
        if self.before is None:
            messagebox.showinfo("Run audit first", "Run the audit before applying a fix.")
            return
        self.after = self.run(True)
        self.report(f"AFTER REWEIGHING ({self.group.get()})", self.after)
        self.draw(charts.selection_bar(self.before["rates"], self.after["rates"], self.group.get()))
        self.say("Fix applied. Compare the red and green bars.")

    def show_chart(self):
        if not self.need_data():
            return
        g, c = self.group.get(), self.chart_choice.get()
        if c.startswith("Approval"):
            if self.before is None:
                return self.audit()
            fig = charts.selection_bar(self.before["rates"], self.after["rates"] if self.after else None, g)
        elif c.startswith("Heatmap"):
            if len(self.cat_cols) < 2:
                return messagebox.showinfo("Heatmap", "Needs two categorical columns.")
            fig = charts.heatmap(self.df, self.cat_cols[0], self.cat_cols[1])
        elif c.startswith("Violin"):
            fig = charts.violin(self.df, g, self.numeric[0])
        elif c.startswith("Pie"):
            fig = charts.pie(self.df, g)
        else:
            fig = charts.correlation(self.df)
        self.draw(fig)

    def show_plotly(self):
        if self.need_data():
            path = os.path.join(tempfile.gettempdir(), "bias_auditor_dashboard.html")
            charts.plotly_dashboard(self.df, self.group.get(), self.numeric, path)
            webbrowser.open("file://" + path)
            self.say("Plotly dashboard opened in your browser.")

    # ------------------------------------------------------------ files
    def save_csv(self):
        if self.need_data():
            p = filedialog.asksaveasfilename(defaultextension=".csv")
            if p:
                self.df.to_csv(p, index=False)

    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not p:
            return
        df = pd.read_csv(p)
        if "approved" not in df.columns:
            return messagebox.showerror("Missing column", "CSV needs an 'approved' column with 0/1 values.")
        cats = [c for c in df.select_dtypes(exclude="number") if df[c].nunique() <= 6]
        nums = [c for c in df.select_dtypes("number") if c != "approved"]
        if not cats or not nums:
            return messagebox.showerror("Unsuitable CSV", "Need at least one text column (e.g. gender) and one numeric column.")
        self.df, filled = an.clean(df)
        self.numeric, self.cat_cols = nums, cats
        self.group.config(values=cats)
        self.group.current(0)
        self.before = self.after = None
        self.log(f"[Data] Loaded {os.path.basename(p)}: {len(df)} rows, protected candidates {cats}")
        self.say("CSV loaded. Run the audit.")

    # ------------------------------------------------------------ experiments
    def e1a(self):
        tp = ex.twin_primes(1000)
        self.log(f"[Exp 1a] {len(tp)} twin prime pairs below 1000:\n  " + ", ".join(f"({a},{b})" for a, b in tp))

    def e1b(self):
        n = simpledialog.askinteger("Permutations & combinations", "n =", parent=self.root, minvalue=0, maxvalue=170)
        r = simpledialog.askinteger("Permutations & combinations", "r =", parent=self.root, minvalue=0, maxvalue=n) if n is not None else None
        if n is not None and r is not None:
            self.log(f"[Exp 1b] p({n},{r}) = {ex.permutations(n, r)} | c({n},{r}) = {ex.combinations(n, r)}")

    def e2a(self):
        self.log("[Exp 2a]\n" + ex.list_operations())

    def e2b(self):
        users = {101: {"name": "Asha", "age": 21}, 102: {"name": "Ravi", "age": 23}, 103: {"name": "Maya", "age": 22}}
        uid = simpledialog.askinteger("User lookup", f"Enter user id {list(users)}:", parent=self.root)
        if uid is not None:
            self.log("[Exp 2b] " + ex.find_user(users, uid))

    def e3(self):
        roll = simpledialog.askstring("Student", "Roll no:", parent=self.root)
        name = simpledialog.askstring("Student", "Name:", parent=self.root) if roll else None
        cls = simpledialog.askstring("Student", "Class:", parent=self.root) if name else None
        if cls:
            self.log("[Exp 3] " + str(ex.Student(roll, name, cls)))

    def e4a(self):
        self.log("[Exp 4a]\n" + ex.swap_columns_demo())

    def e4b(self):
        self.log("[Exp 4b]\n" + ex.sales_filter_demo())

    def how_to(self):
        messagebox.showinfo("How to use",
                            "1. Generate data (Ctrl+G)\n2. Run audit (Ctrl+R): disparate impact below 0.80 means biased\n"
                            "3. Apply reweighing fix (Ctrl+F) and compare\n4. Untick 'Model can see gender/region' "
                            "and audit again: bias remains because income is a proxy\n\n"
                            "Right-click the report for copy/clear. Experiments menu holds the syllabus programs.")


if __name__ == "__main__":
    root = tk.Tk()
    BiasAuditorApp(root)
    root.mainloop()
