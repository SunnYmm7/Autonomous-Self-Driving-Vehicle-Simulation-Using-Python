"""
simulation/simulation.py – Main simulation loop.
Polished HUD, speed multiplier, fitness graph, best-car highlight, trail.
"""
import pygame
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from config import *
from car.car                  import Car
from car.sensors              import Sensors
from ai.neural_network        import NeuralNetwork
from genetic.population       import Genome, evolve, load_best_genome, create_population_from_saved
from environment.track        import Track
from simulation.fitness       import calculate_fitness


# ─────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────
def _draw_rounded_box(surf, rect, color, radius=8, border=None, bw=1):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, bw, border_radius=radius)


def _text(surf, font, msg, pos, color=None, align="left"):
    color = color or COL_TEXT
    img = font.render(msg, True, color)
    x, y = pos
    if align == "center":
        x -= img.get_width() // 2
    elif align == "right":
        x -= img.get_width()
    surf.blit(img, (x, y))


# ─────────────────────────────────────────────
#  Button
# ─────────────────────────────────────────────
class Button:
    def __init__(self, rect, label, col, hover_col):
        self.rect      = rect
        self.label     = label
        self.col       = col
        self.hover_col = hover_col
        self._hover    = False

    def update(self, mp):
        self._hover = self.rect.collidepoint(mp)

    def draw(self, surf, font):
        _draw_rounded_box(surf, self.rect,
                          self.hover_col if self._hover else self.col,
                          radius=6, border=(80, 85, 110), bw=1)
        _text(surf, font, self.label,
              (self.rect.centerx, self.rect.centery - 7),
              COL_TEXT, align="center")

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


# ─────────────────────────────────────────────
#  Simulation
# ─────────────────────────────────────────────
class Simulation:
    _PW = 220          # HUD panel width
    _GH = 110          # fitness graph height

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Self-Driving Car – Neuro-Evolution")
        self.clock = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_md = pygame.font.SysFont("Segoe UI", 15)
        self.font_sm = pygame.font.SysFont("Segoe UI", 13)

        self.track         = Track()
        self.nn            = NeuralNetwork()
        self.sensors       = Sensors(SENSOR_COUNT)

        self.generation    = 1
        self.paused        = False
        self.speed_mult    = 1
        self.ticks         = 0
        self.all_time_best = 0.0
        self.best_history: list[float] = []
        self.avg_history:  list[float] = []
        self.min_history:  list[float] = []
        self.status_msg    = ""
        self.status_timer  = 0

        # Buttons
        px = SCREEN_WIDTH - self._PW + 10
        self.btn_pause = Button(pygame.Rect(px,       14, 95, 32), "⏸  Pause",
                                (55, 55, 85),  (85, 85, 125))
        self.btn_speed = Button(pygame.Rect(px + 105, 14, 95, 32), "▶▶  1×",
                                (35, 75, 115), (55, 105, 155))
        self.btn_load  = Button(pygame.Rect(px,       54, 95, 32), "📂  Load",
                                (35, 100, 60), (55, 140, 85))
        self.btn_save  = Button(pygame.Rect(px + 105, 54, 95, 32), "💾  Save",
                                (100, 75, 25), (140, 105, 35))
        self.btn_quit  = Button(pygame.Rect(px,       94, 200, 32), "⏹  Quit & Show Stats",
                                (110, 30, 30), (160, 45, 45))

        self.population = [Genome() for _ in range(POPULATION_SIZE)]
        self._reset()

    # ── Reset ─────────────────────────────────────────────────────────
    def _reset(self):
        self.cars  = [Car(220, 500, g) for g in self.population]
        for car in self.cars:
            car.angle = 90.0
        self.ticks = 0

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self):
        while True:
            mp = pygame.mouse.get_pos()
            for b in (self.btn_pause, self.btn_speed,
                      self.btn_load,  self.btn_save, self.btn_quit):
                b.update(mp)

            for event in pygame.event.get():
                self._handle(event)

            steps = self.speed_mult if not self.paused else 0
            for _ in range(steps):
                self._update()

            self._draw()
            self.clock.tick(FPS)

    # ── Events ────────────────────────────────────────────────────────
    def _handle(self, event):
        if event.type == pygame.QUIT:
            self._save_quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self._toggle_pause()
            if event.key == pygame.K_s:
                self._save()

        if self.btn_pause.clicked(event):
            self._toggle_pause()

        if self.btn_speed.clicked(event):
            self.speed_mult = {1: 2, 2: 4, 4: 1}[self.speed_mult]
            self.btn_speed.label = f"▶▶  {self.speed_mult}×"

        if self.btn_load.clicked(event):
            saved = load_best_genome()
            if saved:
                self.population = create_population_from_saved(saved)
                self.generation  = 1
                self._reset()
                self._status("✅  Champion genome loaded!")
            else:
                self._status("⚠   No saved genome found.")

        if self.btn_save.clicked(event):
            self._save()
        if self.btn_quit.clicked(event):
            self._save_quit()

    def _toggle_pause(self):
        self.paused = not self.paused
        self.btn_pause.label = "▶  Resume" if self.paused else "⏸  Pause"

    # ── Simulation step ───────────────────────────────────────────────
    def _update(self):
        self.ticks += 1
        alive = 0

        for car in self.cars:
            if not car.alive:
                continue

            car.sensors = self.sensors.get_readings(
                car.pos, car.angle, self.track.mask)
            action = self.nn.predict(car.get_inputs(), car.genome.weights)
            car.update(action)

            try:
                if self.track.mask.get_at((int(car.pos.x), int(car.pos.y))):
                    car.alive          = False
                    car.genome.fitness = calculate_fitness(car)
            except IndexError:
                car.alive          = False
                car.genome.fitness = calculate_fitness(car)

            alive += 1

        # Trail for best car
        best = self._best()
        if best:
            best.add_trail_point()

        # Time-limit guard
        if self.ticks >= MAX_TICKS_PER_GEN:
            for car in self.cars:
                if car.alive:
                    car.alive          = False
                    car.genome.fitness = calculate_fitness(car)
            alive = 0

        if alive == 0:
            self._end_gen()

    def _end_gen(self):
        fits   = [g.fitness for g in self.population]
        best_f = max(fits) if fits else 0.0
        avg_f  = (sum(fits) / len(fits)) if fits else 0.0

        self.best_history.append(best_f)
        self.avg_history.append(avg_f)
        self.min_history.append(min(fits) if fits else 0.0)
        if best_f > self.all_time_best:
            self.all_time_best = best_f

        self.population = evolve(self.population)
        self.generation += 1
        self._reset()

    def _best(self):
        alive = [c for c in self.cars if c.alive]
        return max(alive, key=lambda c: c.distance) if alive else None

    # ── Drawing ───────────────────────────────────────────────────────
    def _draw(self):
        self.track.draw(self.screen)

        best = self._best()
        for car in self.cars:
            car.draw(self.screen, is_best=(car is best))

        self._draw_hud()

        if self.status_timer > 0:
            self._draw_toast()
            self.status_timer -= 1

        pygame.display.flip()

    def _draw_hud(self):
        surf = self.screen
        pw   = self._PW
        px   = SCREEN_WIDTH - pw

        # Panel
        panel = pygame.Surface((pw, SCREEN_HEIGHT), pygame.SRCALPHA)
        panel.fill((14, 15, 26, 215))
        surf.blit(panel, (px, 0))
        pygame.draw.line(surf, (45, 50, 78), (px, 0), (px, SCREEN_HEIGHT), 2)

        # Buttons
        for b in (self.btn_pause, self.btn_speed, self.btn_load, self.btn_save, self.btn_quit):
            b.draw(surf, self.font_sm)

        # ── Stats ─────────────────────────────────────────────────────
        sy = 140
        _draw_rounded_box(surf, pygame.Rect(px + 8, sy, pw - 16, 150),
                          (20, 23, 38), radius=8, border=(42, 48, 72))

        best_car  = self._best()
        alive_cnt = sum(1 for c in self.cars if c.alive)
        best_dist = max((c.distance for c in self.cars), default=0.0)
        spd       = round(best_car.speed, 1) if best_car else 0

        rows = [
            ("Generation",   str(self.generation)),
            ("Alive",        f"{alive_cnt} / {POPULATION_SIZE}"),
            ("Tick",         f"{self.ticks} / {MAX_TICKS_PER_GEN}"),
            ("Best dist",    f"{best_dist:.0f} px"),
            ("Speed",        f"{spd} px/f"),
            ("All-time best",f"{self.all_time_best:.0f}"),
        ]
        for i, (label, val) in enumerate(rows):
            ry = sy + 12 + i * 23
            _text(surf, self.font_sm, label, (px + 14, ry), COL_TEXT_DIM)
            _text(surf, self.font_md, val,
                  (SCREEN_WIDTH - 12, ry), COL_TEXT, align="right")

        # ── Fitness graph ──────────────────────────────────────────────
        gy = sy + 162
        gh = self._GH
        gw = pw - 16

        _draw_rounded_box(surf, pygame.Rect(px + 8, gy, gw, gh + 26),
                          (20, 23, 38), radius=8, border=(42, 48, 72))
        _text(surf, self.font_sm, "Fitness / Generation",
              (px + 14, gy + 6), COL_TEXT_DIM)

        if len(self.best_history) > 1:
            bh = self.best_history[-40:]
            ah = self.avg_history[-40:]
            mx  = max(bh) or 1
            n   = len(bh)
            gx0 = px + 12
            gy0 = gy + 22

            def pt(idx, val):
                rx = gx0 + idx * (gw - 8) // (n - 1)
                ry = gy0 + gh - int(val / mx * (gh - 4))
                return rx, ry

            avg_pts  = [pt(i, v) for i, v in enumerate(ah)]
            best_pts = [pt(i, v) for i, v in enumerate(bh)]

            pygame.draw.lines(surf, (55, 110, 170), False, avg_pts,  1)
            pygame.draw.lines(surf, COL_BEST_CAR,   False, best_pts, 2)
            pygame.draw.circle(surf, COL_BEST_CAR, best_pts[-1], 3)

            # Legend inside graph
            lx = px + 14
            ly = gy + gh + 10
            pygame.draw.line(surf, COL_BEST_CAR,   (lx, ly + 4), (lx + 16, ly + 4), 2)
            _text(surf, self.font_sm, "Best", (lx + 20, ly - 1), COL_TEXT_DIM)
            pygame.draw.line(surf, (55, 110, 170), (lx + 60, ly + 4), (lx + 76, ly + 4), 1)
            _text(surf, self.font_sm, "Avg",  (lx + 80, ly - 1), COL_TEXT_DIM)

        # ── Car legend ────────────────────────────────────────────────
        ly2 = gy + gh + 42
        _draw_rounded_box(surf, pygame.Rect(px + 8, ly2, pw - 16, 58),
                          (20, 23, 38), radius=8, border=(42, 48, 72))
        _text(surf, self.font_sm, "Car colours", (px + 14, ly2 + 6), COL_TEXT_DIM)

        legends = [
            (COL_BEST_CAR,  "Best this gen"),
            (COL_CAR_ALIVE, "Alive"),
            (COL_CAR_DEAD,  "Dead"),
        ]
        for j, (c, lbl) in enumerate(legends):
            row_y = ly2 + 20 + j * 13
            pygame.draw.circle(surf, c, (px + 18, row_y + 4), 4)
            _text(surf, self.font_sm, lbl, (px + 28, row_y), COL_TEXT)

        # ── Key hints ─────────────────────────────────────────────────
        hy = SCREEN_HEIGHT - 50
        for i, h in enumerate(["[Space] Pause / Resume", "[S] Save genome"]):
            _text(surf, self.font_sm, h, (px + 12, hy + i * 16), COL_TEXT_DIM)

    def _draw_toast(self):
        img  = self.font_md.render(self.status_msg, True, (15, 15, 25))
        w, h = img.get_size()
        bx   = SCREEN_WIDTH // 2 - w // 2 - 12
        by   = SCREEN_HEIGHT - 52
        _draw_rounded_box(self.screen,
                          pygame.Rect(bx, by, w + 24, h + 14),
                          (210, 225, 255), radius=7)
        self.screen.blit(img, (bx + 12, by + 7))

    # ── Save / Quit ───────────────────────────────────────────────────
    def _save(self):
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        from genetic.population import save_genome
        save_genome(self.population[0])
        self._status("💾  Best genome saved!")

    def _save_quit(self):
        self._save()
        pygame.quit()
        self._show_stats()
        sys.exit()

    def _show_stats(self):
        n = len(self.best_history)
        if n < 2:
            return
        gens = list(range(1, n + 1))
        best = np.array(self.best_history)
        avg  = np.array(self.avg_history)
        mn   = np.array(self.min_history)

        DARK=  "#0f0f19"; PANEL= "#14151e"; GOLD= "#FFDC00"
        BLUE=  "#3778d6"; RED=   "#e04040"; GREY= "#787890"; WHITE= "#dce1f0"
        plt.rcParams.update({"figure.facecolor":DARK,"axes.facecolor":PANEL,
            "axes.edgecolor":"#2d3250","axes.labelcolor":WHITE,
            "xtick.color":GREY,"ytick.color":GREY,"text.color":WHITE,
            "grid.color":"#1e2035","grid.linewidth":0.8})

        # ── Figure 1: Charts ──────────────────────────────────────────
        fig = plt.figure(figsize=(13,8),facecolor=DARK)
        fig.suptitle(f"AI Self-Driving Car  ·  {n} Generations",
                     color=WHITE,fontsize=15,fontweight="bold",y=0.97)
        gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35,
                               left=0.07,right=0.97,top=0.91,bottom=0.09)

        ax1=fig.add_subplot(gs[0,:2])
        ax1.fill_between(gens,mn,best,alpha=0.12,color=GOLD)
        ax1.plot(gens,mn,color=RED,lw=1,linestyle="--",label="Min")
        ax1.plot(gens,avg,color=BLUE,lw=1.5,label="Avg")
        ax1.plot(gens,best,color=GOLD,lw=2,label="Best")
        ax1.set_title("Fitness per Generation",color=WHITE,fontsize=11)
        ax1.set_xlabel("Generation"); ax1.set_ylabel("Fitness")
        ax1.legend(facecolor=PANEL,edgecolor="#2d3250",labelcolor=WHITE,fontsize=9)
        ax1.grid(True)

        ax2=fig.add_subplot(gs[0,2])
        delta=np.diff(best,prepend=best[0])
        ax2.bar(gens,delta,color=[GOLD if d>=0 else RED for d in delta],width=0.8)
        ax2.axhline(0,color=GREY,lw=0.8,linestyle="--")
        ax2.set_title("Best Δ / Gen",color=WHITE,fontsize=11)
        ax2.set_xlabel("Generation"); ax2.set_ylabel("Δ Fitness"); ax2.grid(True,axis="y")

        ax3=fig.add_subplot(gs[1,0])
        ratio=np.where(best>0,avg/best,0)
        ax3.plot(gens,ratio,color=BLUE,lw=1.5)
        ax3.fill_between(gens,ratio,alpha=0.15,color=BLUE)
        ax3.set_ylim(0,1.05)
        ax3.set_title("Avg/Best Ratio",color=WHITE,fontsize=10)
        ax3.set_xlabel("Generation"); ax3.grid(True)

        ax4=fig.add_subplot(gs[1,1])
        rb=np.maximum.accumulate(best)
        ax4.plot(gens,rb,color=GOLD,lw=2)
        ax4.fill_between(gens,rb,alpha=0.15,color=GOLD)
        ax4.set_title("All-Time Best",color=WHITE,fontsize=11)
        ax4.set_xlabel("Generation"); ax4.grid(True)

        ax5=fig.add_subplot(gs[1,2]); ax5.axis("off")
        best_gen=int(np.argmax(best))+1
        first_90=next((i+1 for i,v in enumerate(rb) if v>=0.9*rb[-1]),n)
        for k,(lbl,val) in enumerate([
            ("Generations",str(n)),("Peak fitness",f"{best.max():.1f}"),
            ("Peak at gen",str(best_gen)),("Final avg",f"{avg[-1]:.1f}"),
            ("90% at gen",f"gen {first_90}"),
            ("Avg gain",f"{delta[delta>0].mean():.2f}" if (delta>0).any() else "—"),
        ]):
            y=0.88-k*0.13
            ax5.text(0.02,y,lbl+":",transform=ax5.transAxes,color=GREY,fontsize=9,va="top")
            ax5.text(0.98,y,val,transform=ax5.transAxes,color=WHITE,fontsize=9,va="top",ha="right",fontweight="bold")
        ax5.set_title("Summary",color=WHITE,fontsize=11)
        ax5.set_facecolor(PANEL)

        plt.tight_layout()
        try:
            plt.show(block=True)
        except Exception:
            plt.savefig("stats_report.png",dpi=120,bbox_inches="tight",facecolor=DARK)
            print("Saved stats_report.png")

        # ── Figure 2: Tables ──────────────────────────────────────────
        self._show_tables(n, gens, best, avg, mn, delta, rb)

    def _show_tables(self, n, gens, best, avg, mn, delta, rb):
        DARK=  "#0f0f19"; PANEL= "#14151e"; GOLD= "#FFDC00"
        BLUE=  "#3778d6"; RED=   "#e04040"; GREY= "#787890"; WHITE= "#dce1f0"
        ACCENT= "#1e2540"

        fig2 = plt.figure(figsize=(14, 9), facecolor=DARK)
        fig2.suptitle("AI Self-Driving Car  ·  Data Tables",
                      color=WHITE, fontsize=15, fontweight="bold", y=0.97)

        gs2 = gridspec.GridSpec(2, 1, figure=fig2,
                                hspace=0.55, top=0.91, bottom=0.04,
                                left=0.04, right=0.96)

        # ── Table 1: Per-generation stats (sampled if > 30 rows) ──────
        ax_t1 = fig2.add_subplot(gs2[0])
        ax_t1.axis("off")
        ax_t1.set_title("Per-Generation Statistics", color=WHITE,
                         fontsize=12, fontweight="bold", pad=10)

        # Sample rows evenly if many generations (max 25 rows)
        max_rows = 25
        if n <= max_rows:
            indices = list(range(n))
        else:
            step = n / max_rows
            indices = sorted(set(int(i * step) for i in range(max_rows)))
            indices[-1] = n - 1   # always include last gen

        rows = []
        for i in indices:
            g = i + 1
            d = delta[i]
            rows.append([
                str(g),
                f"{best[i]:.1f}",
                f"{avg[i]:.1f}",
                f"{mn[i]:.1f}",
                f"{best[i]-avg[i]:.1f}",
                f"{'+' if d>=0 else ''}{d:.1f}",
                f"{rb[i]:.1f}",
            ])

        col_labels = ["Gen", "Best", "Avg", "Min", "Spread", "Δ Best", "All-Time Best"]
        col_widths  = [0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.18]

        # Cell colours: alternate rows, highlight peak gen
        best_gen_idx_in_rows = None
        peak_val = best.max()
        cell_colors = []
        for r_idx, row in enumerate(rows):
            row_c = []
            is_peak = float(row[1]) == peak_val
            if is_peak:
                best_gen_idx_in_rows = r_idx
            for c_idx in range(len(col_labels)):
                if is_peak:
                    row_c.append("#2a3a0a")   # golden-green highlight for peak row
                elif r_idx % 2 == 0:
                    row_c.append(ACCENT)
                else:
                    row_c.append(PANEL)
            cell_colors.append(row_c)

        tbl1 = ax_t1.table(
            cellText=rows,
            colLabels=col_labels,
            colWidths=col_widths,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
        )
        tbl1.auto_set_font_size(False)
        tbl1.set_fontsize(8.5)
        tbl1.scale(1, 1.4)

        # Style header row
        for c in range(len(col_labels)):
            cell = tbl1[0, c]
            cell.set_facecolor("#1a2a5e")
            cell.set_text_props(color=GOLD, fontweight="bold")
            cell.set_edgecolor("#2d3250")

        # Style data cells
        for r in range(1, len(rows)+1):
            for c in range(len(col_labels)):
                cell = tbl1[r, c]
                cell.set_edgecolor("#2d3250")
                # Colour delta column: green=gain, red=loss
                if c == 5:
                    val_str = rows[r-1][c]
                    val = float(val_str)
                    cell.set_text_props(
                        color="#55ee55" if val > 0 else ("#ee5555" if val < 0 else GREY)
                    )
                else:
                    cell.set_text_props(color=WHITE)

        # ── Table 2: Milestone summary ────────────────────────────────
        ax_t2 = fig2.add_subplot(gs2[1])
        ax_t2.axis("off")
        ax_t2.set_title("Run Milestones & Summary", color=WHITE,
                         fontsize=12, fontweight="bold", pad=10)

        best_gen   = int(np.argmax(best)) + 1
        worst_gen  = int(np.argmin(best)) + 1
        first_50   = next((i+1 for i,v in enumerate(rb) if v >= rb[-1]*0.50), n)
        first_75   = next((i+1 for i,v in enumerate(rb) if v >= rb[-1]*0.75), n)
        first_90   = next((i+1 for i,v in enumerate(rb) if v >= rb[-1]*0.90), n)
        first_100  = next((i+1 for i,v in enumerate(rb) if v >= rb[-1]*1.00), n)
        pos_gens   = int((delta > 0).sum())
        neg_gens   = int((delta < 0).sum())
        flat_gens  = int((delta == 0).sum())
        avg_gain   = float(delta[delta > 0].mean()) if (delta > 0).any() else 0.0
        avg_loss   = float(delta[delta < 0].mean()) if (delta < 0).any() else 0.0
        streak     = 0; cur = 0
        for d in delta:
            if d > 0: cur += 1; streak = max(streak, cur)
            else: cur = 0

        milestone_rows = [
            ["Total generations",        str(n),          "Positive Δ gens",    str(pos_gens)],
            ["Peak fitness",             f"{best.max():.2f}", "Negative Δ gens", str(neg_gens)],
            ["Peak at generation",       str(best_gen),   "Flat gens (Δ=0)",    str(flat_gens)],
            ["Worst gen best fitness",   f"{best.min():.2f}", "Avg gain / improving gen", f"{avg_gain:.2f}"],
            ["Worst at generation",      str(worst_gen),  "Avg loss / declining gen", f"{avg_loss:.2f}"],
            ["Final best fitness",       f"{best[-1]:.2f}", "Longest improve streak", str(streak)+" gens"],
            ["Final avg fitness",        f"{avg[-1]:.2f}","50% threshold reached",  f"gen {first_50}"],
            ["Final min fitness",        f"{mn[-1]:.2f}", "75% threshold reached",  f"gen {first_75}"],
            ["Overall improvement",      f"{best[-1]-best[0]:.2f}", "90% threshold reached", f"gen {first_90}"],
            ["All-time best",            f"{rb[-1]:.2f}", "100% threshold reached", f"gen {first_100}"],
        ]

        m_col_labels  = ["Metric", "Value", "Metric", "Value"]
        m_col_widths  = [0.28, 0.16, 0.28, 0.16]

        m_cell_colors = []
        for r_idx, row in enumerate(milestone_rows):
            bg = ACCENT if r_idx % 2 == 0 else PANEL
            m_cell_colors.append([bg]*4)

        tbl2 = ax_t2.table(
            cellText=milestone_rows,
            colLabels=m_col_labels,
            colWidths=m_col_widths,
            cellLoc="center",
            loc="center",
            cellColours=m_cell_colors,
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8.5)
        tbl2.scale(1, 1.45)

        for c in range(4):
            cell = tbl2[0, c]
            cell.set_facecolor("#1a2a5e")
            cell.set_text_props(color=GOLD, fontweight="bold")
            cell.set_edgecolor("#2d3250")

        for r in range(1, len(milestone_rows)+1):
            for c in range(4):
                cell = tbl2[r, c]
                cell.set_edgecolor("#2d3250")
                # Value columns in bright white, label cols in grey
                cell.set_text_props(color=WHITE if c%2==1 else GREY)

        plt.tight_layout()
        try:
            plt.show(block=True)
        except Exception:
            plt.savefig("tables_report.png", dpi=120, bbox_inches="tight", facecolor=DARK)
            print("Saved tables_report.png")

    def _status(self, msg, duration=180):
        self.status_msg   = msg
        self.status_timer = duration