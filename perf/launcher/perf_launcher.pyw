"""Performance Lab launcher window (Windows; also runs on Linux/macOS with Tk).

Start it with perf\\perf.bat (it installs Python and the packages first if needed), the desktop shortcut, or
`python perf/launcher/perf_launcher.pyw`. All work happens in perf/launcher/core.py; this file is only the window.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

RED, INK, MUTED, GOOD, BAD, BG = "#c8102e", "#0c1118", "#5c6570", "#0b8a3e", "#c8102e", "#f9fafb"


class Launcher(tk.Tk):
    def __init__(self, auto_start: bool | None = None):
        super().__init__()
        self.settings = core.load_settings()
        self.title("Performance Lab - Lions Racing")
        self.configure(bg=BG)
        self.minsize(640, 560)
        try:
            if core.WINDOWS:
                self.iconbitmap(str(core.ICON))
        except tk.TclError:
            pass
        self.q: queue.Queue = queue.Queue()            # work threads -> window
        self.busy = False
        self.running = False
        self._style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._pump)
        self.after(300, self._status_loop)
        self.say("Welcome. Press \"Start and open\" to use the Performance Lab in your web browser.")
        start = self.settings["auto_start"] if auto_start is None else auto_start
        self.after(600, self.first_checks if not start else self.start_and_open)

    # ------------------------------------------------------------------ layout
    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("vista" if core.WINDOWS else "clam")
        except tk.TclError:
            pass
        s.configure("TFrame", background=BG)
        s.configure("Card.TLabelframe", background=BG, padding=10)
        s.configure("Card.TLabelframe.Label", background=BG, foreground=INK, font=("Segoe UI", 10, "bold"))
        s.configure("TLabel", background=BG, foreground=INK, font=("Segoe UI", 10))
        s.configure("Muted.TLabel", foreground=MUTED, font=("Segoe UI", 9))
        s.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 8))
        s.configure("TButton", font=("Segoe UI", 10), padding=(10, 5))
        s.configure("TCheckbutton", background=BG, font=("Segoe UI", 9))

    def _build(self):
        top = tk.Frame(self, bg="white", highlightthickness=0)
        top.pack(fill="x")
        try:
            self.logo = tk.PhotoImage(file=str(core.HERE / "logo.png"))
            tk.Label(top, image=self.logo, bg="white").pack(side="left", padx=(16, 10), pady=8)
        except tk.TclError:
            pass
        tk.Label(top, text="PERFORMANCE LAB", bg="white", fg=INK, font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Frame(self, bg=RED, height=3).pack(fill="x")

        body = ttk.Frame(self, padding=(16, 12))
        body.pack(fill="both", expand=True)

        # status + main buttons
        st = ttk.Frame(body)
        st.pack(fill="x")
        self.dot = tk.Canvas(st, width=16, height=16, bg=BG, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 8))
        self.dot_id = self.dot.create_oval(2, 2, 14, 14, fill=MUTED, outline="")
        self.status_lbl = ttk.Label(st, text="Checking ...", font=("Segoe UI", 11, "bold"))
        self.status_lbl.pack(side="left")
        btns = ttk.Frame(body)
        btns.pack(fill="x", pady=(10, 4))
        self.b_start = ttk.Button(btns, text="▶  Start and open", style="Big.TButton", command=self.start_and_open)
        self.b_start.pack(side="left")
        self.b_open = ttk.Button(btns, text="Open in the browser", command=self.open_browser)
        self.b_open.pack(side="left", padx=8)
        self.b_stop = ttk.Button(btns, text="■  Stop", command=self.stop)
        self.b_stop.pack(side="left")
        ttk.Label(body, text="The Lab runs in your web browser. You can close the browser tab at any time; "
                             "the Lab keeps running until you press Stop or close this window.",
                  style="Muted.TLabel", wraplength=600).pack(anchor="w", pady=(2, 10))

        row = ttk.Frame(body)
        row.pack(fill="both", expand=True)
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)

        # health check
        hc = ttk.LabelFrame(row, text="Is everything ready?", style="Card.TLabelframe")
        hc.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.checks_box = ttk.Frame(hc)
        self.checks_box.pack(fill="x")
        ttk.Label(self.checks_box, text="Not checked yet.", style="Muted.TLabel").pack(anchor="w")
        hb = ttk.Frame(hc)
        hb.pack(fill="x", pady=(8, 0))
        self.b_check = ttk.Button(hb, text="Check again", command=self.run_checks)
        self.b_check.pack(side="left")
        self.b_repair = ttk.Button(hb, text="Repair", command=self.repair, state="disabled")
        self.b_repair.pack(side="left", padx=6)

        # data and help
        side = ttk.Frame(row)
        side.grid(row=0, column=1, sticky="nsew")
        data = ttk.LabelFrame(side, text="Your data (saved setups)", style="Card.TLabelframe")
        data.pack(fill="x")
        for text, cmd in (("Back up setups ...", self.do_backup), ("Restore from a backup ...", self.do_restore),
                          ("Open the setups folder", lambda: core.open_path(str(core.CONFIGS)))):
            ttk.Button(data, text=text, command=cmd).pack(fill="x", pady=2)
        hlp = ttk.LabelFrame(side, text="Help", style="Card.TLabelframe")
        hlp.pack(fill="x", pady=(8, 0))
        for text, cmd in (("Open the user guide", self.open_guide),
                          ("Read the technical paper (PDF)", self.open_paper),
                          ("Save a help report ...", self.do_report),
                          ("Put a shortcut on the desktop", self.do_shortcut)):
            ttk.Button(hlp, text=text, command=cmd).pack(fill="x", pady=2)

        # settings
        sett = ttk.Frame(body)
        sett.pack(fill="x", pady=(10, 4))
        self.v_auto = tk.BooleanVar(value=self.settings["auto_start"])
        self.v_stop = tk.BooleanVar(value=self.settings["stop_on_close"])
        ttk.Checkbutton(sett, text="Start the Lab when this window opens", variable=self.v_auto,
                        command=self.save_settings).pack(side="left")
        ttk.Checkbutton(sett, text="Stop it when this window closes", variable=self.v_stop,
                        command=self.save_settings).pack(side="left", padx=12)
        ttk.Label(sett, text="Port").pack(side="left", padx=(12, 4))
        self.v_port = tk.StringVar(value=str(self.settings["port"]))
        e = ttk.Entry(sett, textvariable=self.v_port, width=6)
        e.pack(side="left")
        e.bind("<FocusOut>", lambda _e: self.save_settings())

        # messages
        lf = ttk.LabelFrame(body, text="Messages", style="Card.TLabelframe")
        lf.pack(fill="both", expand=True, pady=(8, 0))
        self.msg = tk.Text(lf, height=7, wrap="word", font=("Consolas" if core.WINDOWS else "Courier", 9),
                           bg="white", fg=INK, relief="flat", state="disabled")
        self.msg.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ helpers
    @property
    def port(self) -> int:
        return int(self.settings["port"])

    def say(self, text: str):
        core.log(text)
        self.msg.configure(state="normal")
        self.msg.insert("end", text + "\n")
        self.msg.see("end")
        self.msg.configure(state="disabled")

    def work(self, label: str, fn, done=None):
        """Run fn() on a thread; done(result) runs in the window afterwards."""
        if self.busy:
            self.say("Please wait, still busy ...")
            return
        self.busy = True
        self.config(cursor="watch")
        self.status_lbl.configure(text=label)

        def run():
            try:
                res = fn()
                err = None
            except Exception as e:  # noqa: BLE001
                res, err = None, e
            self.q.put(("done", done, res, err))

        threading.Thread(target=run, daemon=True).start()

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "say":
                    self.say(item[1])
                elif item[0] == "status":
                    self.show_status(item[1])
                elif item[0] == "done":
                    _, done, res, err = item
                    self.busy = False
                    self.config(cursor="")
                    if err is not None:
                        self.say(f"Something went wrong: {err}")
                    elif done:
                        done(res)
                    self.refresh_status()
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def say_from_thread(self, text: str):
        self.q.put(("say", text))

    def refresh_status(self):
        """Ask (on a thread) whether the Lab answers; the answer is shown by _pump."""
        threading.Thread(target=lambda: self.q.put(("status", core.is_running(self.port, 0.5))), daemon=True).start()

    def _status_loop(self):
        if not self.busy:
            self.refresh_status()
        self.after(3000, self._status_loop)

    def show_status(self, running: bool):
        self.running = running
        if self.busy:
            return
        self.dot.itemconfigure(self.dot_id, fill=GOOD if running else MUTED)
        self.status_lbl.configure(text=f"Running at {core.url(self.port)}" if running else "Stopped")
        self.b_open.configure(state="normal" if running else "disabled")
        self.b_stop.configure(state="normal" if running else "disabled")

    def save_settings(self):
        try:
            port = int(self.v_port.get())
            if not 1024 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Port", "The port must be a number between 1024 and 65535 (default 8090).")
            self.v_port.set(str(self.settings["port"]))
            return
        self.settings.update(port=port, auto_start=self.v_auto.get(), stop_on_close=self.v_stop.get())
        core.save_settings(self.settings)

    # ------------------------------------------------------------------ actions
    def first_checks(self):
        self.run_checks()

    def show_checks(self, checks):
        for w in self.checks_box.winfo_children():
            w.destroy()
        for c in checks:
            f = ttk.Frame(self.checks_box)
            f.pack(fill="x", anchor="w", pady=1)
            mark = ("\u2714" if c.ok else "\u2013" if c.optional else "\u2716") if core.WINDOWS else \
                ("OK" if c.ok else "--" if c.optional else "!!")
            tk.Label(f, text=mark, fg=GOOD if c.ok else MUTED if c.optional else BAD, bg=BG, width=2,
                     font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 4))
            ttk.Label(f, text=f"{c.name}: {c.detail}", wraplength=280).pack(side="left", anchor="w")
            if not c.ok and c.fix:
                ttk.Label(self.checks_box, text="→ " + c.fix, style="Muted.TLabel", wraplength=290).pack(anchor="w", padx=(22, 0))
        self.checks = checks
        self.b_repair.configure(state="normal" if any(c.repairable for c in checks) else "disabled")
        bad = [c for c in checks if not c.ok and not c.optional]
        self.say("All checks passed." if not bad else "Not ready: " + "; ".join(c.name for c in bad))
        return not bad

    def run_checks(self, then=None):
        def done(checks):
            ok = self.show_checks(checks)
            if then:
                then(ok)
        self.work("Checking ...", lambda: core.health_check(self.port), done)

    def repair(self):
        miss = [p for c in getattr(self, "checks", []) for p in c.extra.get("missing", [])]
        if not miss:
            self.run_checks()
            return
        if not messagebox.askyesno("Repair", "Install " + ", ".join(miss) + " for this Python now?\n\n"
                                   "Needs internet and takes a few minutes the first time."):
            return
        self.work("Installing ...", lambda: core.pip_install(miss, self.say_from_thread),
                  lambda ok: (self.say("Installed." if ok else "The installation did not work: see the messages above."),
                              self.run_checks()))

    def start_and_open(self):
        if core.is_running(self.port):
            self.open_browser()
            return

        def after_checks(ok):
            if not ok:
                messagebox.showwarning("Not ready", "Something is missing (see \"Is everything ready?\").\n\n"
                                       "Press Repair if it is active, otherwise follow the hint under the red cross.")
                return
            self.work("Starting ...", lambda: core.start_app(self.port), started)

        def started(res):
            ok, text = res
            self.say(("Started: " if ok else "") + text)
            if ok:
                self.open_browser()
            else:
                messagebox.showerror("Could not start", text)

        self.say("Starting the Performance Lab ...")
        self.run_checks(after_checks)

    def open_browser(self, page: str = ""):
        core.open_path(core.url(self.port) + (f"/#{page}" if page else ""))

    def stop(self):
        self.work("Stopping ...", lambda: core.stop_app(self.port), lambda res: self.say(res[1].capitalize() + "."))

    def open_guide(self):
        if core.is_running(self.port):
            self.open_browser("guide")
        else:
            guide = core.REPO / "docs" / "PERF_APP.md"
            self.say("The guide is part of the Lab: press \"Start and open\", then Guide in the top menu. "
                     f"Opening the short manual ({guide.name}) instead.")
            core.open_path(str(guide))

    def open_paper(self):
        p = core.PAPERS / "paper2_performance_lab.pdf"
        if p.exists():
            core.open_path(str(p))
        else:
            self.say(f"The paper is not in this copy of the project ({p}).")

    def do_backup(self):
        def done(path):
            self.say(f"Backup saved: {path}")
            if messagebox.askyesno("Backup", f"Saved:\n{path}\n\nOpen the folder?"):
                core.open_path(str(path.parent))
        self.work("Backing up ...", core.backup, done)

    def do_restore(self):
        z = filedialog.askopenfilename(title="Choose a Performance Lab backup", initialdir=str(core.documents_dir()),
                                       filetypes=[("Backup", "*.zip")])
        if not z:
            return

        def done(r):
            text = (f"{len(r['added'])} setup(s) added, {len(r['same'])} already there"
                    + (f", {len(r['renamed'])} kept next to yours as \"(restored)\"" if r["renamed"] else "")
                    + (f", {len(r['skipped'])} skipped" if r["skipped"] else "") + ".")
            self.say("Restore: " + text)
            messagebox.showinfo("Restore", text + "\n\nReload the browser page to see them.")
        self.work("Restoring ...", lambda: core.restore(z), done)

    def do_report(self):
        def done(path):
            self.say(f"Help report saved: {path}")
            messagebox.showinfo("Help report", f"Saved:\n{path}\n\nSend this file to whoever helps you. It holds the "
                                "checks and logs, no car data.")
            core.open_path(str(path.parent))
        self.work("Writing the report ...", core.help_report, done)

    def do_shortcut(self):
        self.work("Making shortcuts ...", core.create_shortcuts, lambda res: self.say(res[1]))

    def on_close(self):
        self.save_settings()
        if core.is_running(self.port) and self.settings["stop_on_close"]:
            self.status_lbl.configure(text="Stopping ...")
            self.update_idletasks()
            core.stop_app(self.port)
        self.destroy()


def main():
    auto = None
    if "--no-start" in sys.argv:
        auto = False
    app = Launcher(auto_start=auto)
    app.mainloop()


if __name__ == "__main__":
    main()
