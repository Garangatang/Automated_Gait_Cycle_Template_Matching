import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

"""
Written by Ivan Khimach and Grange Simpson
Version: 2026.03.17
"""

class Adaptive_Template_Matching:
    """
    Sliding-window template matcher with multi-stage gating and optional merging.

    Pipeline (per scan)
    -------------------
    1) Build sliding windows and apply absolute-amplitude and min-std gates.
    2) Z-score each window to align spaces with the z-scored template.
    3) Compute SAD (−Σ|Δ|) in z-space (diagnostics); optional low-amplitude gate.
    4) For surviving candidates, compute Pearson r; require r ≥ `threshold`.
    5) Fit y = a*T + b and compute relative RMS error and flat-region RMS gate.
    6) Cluster-merge accepted indices within ±`cluster_radius` using metric
       {'r','sad','sad_shifted'} to reduce near-duplicates.
    7) Record SAD traces to `sad_history` for QC.

    Attributes
    ----------
    template : np.ndarray (z-scored)
    win : int
        Template window length.
    sad_history : list[np.ndarray]
        SAD traces per scan for diagnostics.

    Update rule
    -----------
    `update_template` averages ALL accepted z-scored windows, then z-scores again.
    """
    def __init__(self, cp_inds: list = [145, 195], template_angls: list = [80, 88], 
                template_len: int = 200, baseline: float = 0.0, template_scaler: float = 0.14, 
                reflect: bool = False):
        """Convert the given angle in degrees to a slope. A slope < 90° will be positive, a slope > 90° will be negative.

        Args:
            cp_inds: Indices for points within the template where the slope changes based on the given angles.
            template_angls: Angles to be used in creating the change in slope between the given change points.
            template_len: Wanted length of the template.
            reflect:Reflect the template about the y-axis, to say mark toe-off rather than heel-strike.
        Returns:
            Slope of the line needed to create the line at the given angle.
        """
        self.template_scaler = template_scaler
        self.template = self.make_dual_angle_template(template_angls, cp_inds, template_len, 
                                                baseline)
        
        # Flipping the template if wanted, say for marking toe-off rather than heel-strike.
        if reflect:
            self.template = self.template[::-1]

        # Save of starting template shape to be used when resetting the template between datasets.
        self.template_save = self.template.copy()

        # Normalizing template shape and initializing other global variables.
        self.reset()

    # ===== Template construction ======================================================
    def _angle_to_slope(self, angle_deg):
        """Convert the given angle in degrees to a slope by calculating the ratio of the vertical change to the horizontal change. 
           Utilizing tan(θ) = sin(θ) / cos(θ) = ∆y / ∆x. Angles < 90° will be a positive slope, angles > 90° will be a negative slope.

        Args:
            angle_deg: Input angle in degress to be used for calculating the wanted slope.
        Returns:
            Slope of the line needed to create the line at the given angle.
        """
        return np.tan(np.deg2rad(angle_deg))
    
    def make_dual_angle_template(self, angle_degr_arr: list, 
                                change_points_arr: list,
                                template_length: int=200,
                                baseline: float = 0.0):
        """Create the piecwise linear function that approximates the wanted template shape based off the given change points and angles.

        Args:
            angle_degr_arr: Array of the wanted slope changes within the template.
            change_points_arr: Array of the indices for a change in slope to occur within the template.
            template_length: Max length of the template.
            baseline: Starting values for the template, i.e. if set to zero, then the beginning values will be zero before the given slopes and change points.
        Returns:
            Template constructed from the given slopes, changepoints, and length.
        """

        for cp in change_points_arr:
            if cp > template_length:
                raise ValueError(f"Change point {cp} not in range (0, {template_length}]")
                return

        # 1. Convert degrees to slopes and apply the direction multiplier
        slopes = [self._angle_to_slope(a) for a in angle_degr_arr]
        
        t = np.zeros(template_length)
        current_t = baseline
        
        # 2. Add total_length to indices to define the final segment's boundary
        break_points = list(change_points_arr) + [template_length]
        
        for i in range(len(slopes)):
            start_idx = break_points[i]
            end_idx = break_points[i+1]
            
            segment_len = end_idx - start_idx
            segment_x = np.arange(segment_len)
            # Calculate y-values starting from where the last segment left off
            segment_t = current_t + (slopes[i] * segment_x)
            
            t[start_idx:end_idx] = segment_t
            
            current_t += slopes[i] * segment_len
                
        return t

    def _corrcoef_safe(self, template: list, data_window: list):
        """Calculating pearson-r correlation coefficient between the template and a data_window to find the data 
            window with the highest correlation to the template.

        Args:
            a: Current template
            b: Window of data to correlate the template to
        Returns:
            0 when either vector has zero variance, else returns correlation coefficient between a and b.
        """
        return 0.0 if template.std() == 0 or data_window.std() == 0 else float(np.corrcoef(template, data_window)[0, 1])

    def _rolling_window_correlation(self, template: list, windowed_data: list):
        """
        Pearson Correlation Coefficient (r) calculated through numpy arrays
        """
        # Center the data (Subtract the mean)
        # We subtract the mean of each window and the mean of the template
        windows_centered = windowed_data - np.mean(windowed_data, axis=1, keepdims=True)
        template_centered = template - np.mean(template)
        
        # 3. Calculate the numerator (Dot product along the window axis)
        numerator = np.sum(windows_centered * template_centered, axis=1)
        
        # 4. Calculate the denominator (Product of the norms)
        windows_norm = np.sqrt(np.sum(windows_centered**2, axis=1))
        template_norm = np.sqrt(np.sum(template_centered**2))
        
        return numerator / (windows_norm * template_norm)

    def _min_max_norm(self, x):
        """Min–max normalization (0–1)."""
        x = np.asarray(x, dtype=float)
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-12)

    def _zscore(self, x):
        """Standard score."""
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / (x.std() + 1e-12)

    def reset(self):
        """ Reset the template shape if inputting 

        """
        self.template = self._min_max_norm(self.template_save.astype(float))
        self.template = self.template * self.template_scaler
        self.template_history = [self.template.copy()]
        self.win = len(self.template)
        self.sad_history = []      
        self.last_nsad = None   # last backend z-space SAD

    # --- visualization-only helpers (do NOT affect backend / gating) -------------------------
    def _moving_average_1d(self, y: np.ndarray, w: int) -> np.ndarray:
        """Odd-length moving average for smoother plots."""
        L = len(y)
        w = max(3, min(int(w), L - 1))
        if w % 2 == 0:
            w += 1
        k = np.ones(w, dtype=float) / w
        return np.convolve(np.asarray(y, float), k, mode="same")

    def plot_template(self, title="Template", raw=False):
        data = self.template if raw else self._min_max_norm(self.template)
        plt.figure(figsize=(8, 3))
        plt.plot(data, c="purple")
        plt.title(title + (" (raw z-scored)" if raw else ""))
        plt.grid(); plt.tight_layout(); plt.show()

    def _plot_sad_legacy_visual(self, sad_vis: np.ndarray, title="−Σ|Δ|  (no shift)", threshold: float = 0.0, figsize: tuple=(14, 2.8)):
        """
        • smoothed legacy −Σ|Δ| (raw windows vs template)
        • dashed red line at 0; y-axis with 0 at top 
        """
        #y = np.minimum(np.asarray(sad_vis, float), 0.0)
        y = np.asarray(sad_vis, float)
        y_min = float(np.min(y)); y_max = float(np.max(y))
        pad_bottom = 0.05 * (abs(y_max - y_min)) if (y_max > y_min) else 1.0
        pad_top = 0.05 * (abs(y_min - y_max)) if (y_min < y_max) else 1.0

        plt.figure(figsize=figsize)
        plt.plot(y, lw=1.25, label="−Σ|Δ| (legacy, smoothed)")
        plt.axhline(-0.5, c="red", ls="--", lw=1.0, label="threshold = -0.5")
        #plt.ylim(y_min - pad_bottom, y_max + 0.02 * max(1.0, abs(y_max)))
        plt.ylim(y_min - pad_bottom, y_max + pad_top)
        plt.title(title)
        plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.show()

    def _debug_plots(self, signal_data: np.ndarray, mark_data: np.ndarray, horiz_axis_line_inds: list, 
                    vert_axis_line_inds: list, title: str, label: object, plot_option: str):
        """Calculating pearson-r correlation coefficient between the template and a data_window to find the data 
            window with the highest correlation to the template.

        Args:
            signal_data:
            mark_data:
            horiz_axis_line_inds:
            vert_axis_line_inds:
            plot_option: Can either be "sad_legacy", "matches", "template_composite"
        Returns:
            
        """
        # Debug: panel 1
        if plot_option == "sad_legacy":
            plt.figure(figsize=(12, 2.2))
            plt.plot(data, lw=.8, c="k", alpha=.7, label="signal")
            for thresh in horiz_axis_line_inds: plt.axhline(thresh, c="purple", ls="--", lw=1, label=f"{int(label*100)}th pct")
            xs = mark_data; ys = data[mark_data] if vert_axis_line_inds.size else np.array([])
            if xs.size:
                plt.scatter(xs, ys, s=3, c="purple", alpha=.6, label="candidates (low-amp)")
            plt.title(title)
            plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

        if plot_option == "matches":
            w = self.win
            plt.figure(figsize=(12, 4))
            plt.plot(self._min_max_norm(data), c="black", alpha=.8)
            for i in vert_axis_line_inds: plt.axvspan(i, i+w-1, color="royalblue", alpha=.25)
            plt.title(f"{title}  final matches (n={len(mark_data)})")
            plt.grid(); plt.tight_layout(); plt.show()

        if plot_option == "template_composite":
            plt.figure(figsize=(10, 5))
            draw_segs = signal_data if len(signal_data) <= 200 else signal_data[:200]
            for s in draw_segs:
                plt.plot(self._min_max_norm(s), c="grey", alpha=0.25)
            plt.plot(self._min_max_norm(self.template), c="blue", lw=2, label=f"{label}")
            plt.title(f"{title}")
            plt.grid(); plt.legend(); plt.tight_layout(); plt.show()


    # --- Core template matching algorithm functions-------------------------
    def scan_matches(self, data,
                     threshold,
                     amp_max,
                     rel_err,
                     flat_err_thresh=0.15,
                     pos_shift=20,
                     sad_thresh=0,
                     min_std=1e-3,
                     # low-amplitude gating
                     enforce_low_amp=True,
                     low_amp_quantile=0.10,
                     low_amp_cover=0.60,
                     # clustering controls
                     cluster_metric="sad",      # {"r","sad","sad_shifted"}
                     cluster_radius=None,       # None → defaults to win
                     # visualization (plots only)
                     vis_smooth=True,
                     vis_smooth_win_factor=0.5, # ~0.5 * template length
                     show_debug=True,
                     debug_verbose=False):

        w, N = self.win, len(data)
        if N < w + 1:
            return np.empty(0, int), np.empty(0, float)
        if cluster_radius is None:
            cluster_radius = w

        # Sliding windows (N-w+1, w)
        win_mat = sliding_window_view(data, w)

        # Absolute amplitude gate
        amp_abs_max = np.max(np.abs(win_mat), axis=1)
        amp_ok = amp_abs_max <= amp_max

        # Per-window z-score to match template space
        #win_mean = win_mat.mean(axis=1, keepdims=True)
        win_std  = win_mat.std(axis=1, keepdims=True) + 1e-12
        std_ok   = (win_std.squeeze() >= min_std)
        #win_norm = (win_mat - win_mean) / win_std
        #win_norm = self._zscore(win_mat)
        win_norm = self._min_max_norm(win_mat)

        #print(self.template)
        nsad = -np.sum(np.abs(win_norm - self.template), axis=1) + pos_shift
        #nsad = -np.sum(np.abs(win_mat - self.template), axis=1)
        self.last_nsad = nsad
        self.sad_history.append(nsad.copy())

        # Optional low-amp gate
        if enforce_low_amp:
            q_low   = np.quantile(data, low_amp_quantile)
            frac_lo = (win_mat <= q_low).mean(axis=1)
            low_ok  = frac_lo >= float(low_amp_cover)
        else:
            low_ok = np.ones(len(nsad), dtype=bool)

        # Candidates by inexpensive gates
        idx_cand = np.nonzero((nsad > sad_thresh) & amp_ok & std_ok & low_ok)[0]
        #idx_cand = np.nonzero((nsad > amp_ok) & std_ok & low_ok)[0]

        # VISUAL SAD : raw windows vs template
        #sad_vis = -np.sum(np.abs(win_mat - self.template), axis=1)
        sad_vis = nsad.copy()
        #sad_vis = np.minimum(sad_vis, 0.0)

        accepted, scores = [], []

        pearson_corr_arr = self._rolling_window_correlation(self.template, win_norm[idx_cand])
        for r, i in zip(pearson_corr_arr, idx_cand):
            seg_norm = win_norm[i]
            #r = self._corrcoef_safe(self.template, seg_norm)
            if r < threshold:
                continue

            A = np.vstack([self.template, np.ones_like(self.template)]).T
            a, b = np.linalg.lstsq(A, seg_norm, rcond=None)[0]
            seg_fit = a * self.template + b

            rel_rms = np.sqrt(((seg_norm - seg_fit)**2).mean()) / (
                      np.sqrt(((seg_fit - seg_fit.mean())**2).mean()) + 1e-12)
            if rel_rms > rel_err:
                continue

            rms_flat = np.sqrt(((seg_norm[:w//2] - seg_fit[:w//2])**2).mean())
            if rms_flat > flat_err_thresh:
                continue

            accepted.append(i); scores.append(r)

        accepted = np.asarray(accepted, int)
        scores   = np.asarray(scores,   float)

                # Debug: panel 1
        if show_debug:
            self._plot_sad_legacy_visual(sad_vis, title="−Σ|Δ|  (no shift)", threshold = sad_thresh)

            if enforce_low_amp:
                self._debug_plots(signal_data = data, mark_data = accepted, horiz_axis_line_inds = [],
                                vert_axis_line_inds = idx_cand, title = "Accepted After RMS Calculations", label = low_amp_quantile, 
                                plot_option = "sad_legacy")

        # Cluster merge 
        if accepted.size > 1:
            order    = np.argsort(accepted)
            acc_sort = accepted[order]
            r_sort   = scores[order]

            if cluster_metric == "r":
                metric_seq = r_sort
            elif cluster_metric == "sad":
                metric_seq = nsad[acc_sort]
            elif cluster_metric == "sad_shifted":
                tmp = nsad.copy()
                if acc_sort.size:
                    tmp[acc_sort] = tmp[acc_sort] #+ pos_shift
                metric_seq = tmp[acc_sort]
            else:
                raise ValueError("cluster_metric must be 'r', 'sad', or 'sad_shifted'.")

            keep = []
            k = 0
            while k < len(acc_sort):
                cluster = [k]
                while k + 1 < len(acc_sort) and acc_sort[k + 1] - acc_sort[k] <= cluster_radius:
                    k += 1; cluster.append(k)
                best_local = cluster[np.argmax(metric_seq[cluster])]
                keep.append(best_local)
                k += 1

            accepted = acc_sort[keep]
            scores   = r_sort[keep]

        if debug_verbose:
            print(f"windows:{N-w+1} | cand:{idx_cand.size} | final:{accepted.size} | cluster_metric={cluster_metric}")

        return accepted, scores

    def update_template(self, data, match_idx, top_n=50, show_debug=True):
        if match_idx.size == 0:
            print("⚠️  No matches – template unchanged.")
            return
        w = self.win
        #segs = [self._zscore(data[i:i + w]) for i in match_idx]
        segs = [data[i:i + w] for i in match_idx]
        #self.template = self._zscore(np.mean(segs, axis=0))
        self.template = self._min_max_norm(np.mean(segs, axis = 0))
        #self.template = self.template * self.template_scaler
        self.template_history.append(self.template.copy())

        if show_debug:
            self._debug_plots(signal_data = segs, mark_data = np.array([]), horiz_axis_line_inds = [], vert_axis_line_inds = [],
                            title = f"Template updated from all accepted segments (n={len(segs)})", label = "New template (ALL)",
                            plot_option = "template_composite")

    """
    One-shot scan with plotting and summary stats.

    Returns
    -------
    idx : np.ndarray[int]
        Accepted window start indices.
    sco : np.ndarray[float]
        Pearson r scores.

    Plot
    ----
    Overlays accepted windows as shaded spans over normalized signal.
    """
    def final_template_match_and_plot(self, name, data,
                                  threshold, amp_max,
                                  pos_shift, sad_thresh,
                                  min_std=1e-3,
                                  rel_err=0.20,
                                  enforce_low_amp=True,
                                  low_amp_quantile=0.10,
                                  low_amp_cover=0.60,
                                  # NEW:
                                  cluster_metric="sad",
                                  cluster_radius=None,
                                  show_debug=True):

        # Scanning data with template for matches
        idx, sco = self.scan_matches(
            data,
            threshold,
            amp_max,
            rel_err,
            pos_shift=pos_shift,
            sad_thresh=sad_thresh,
            min_std=min_std,
            enforce_low_amp=enforce_low_amp,
            low_amp_quantile=low_amp_quantile,
            low_amp_cover=low_amp_cover,
            # ↓↓↓
            cluster_metric=cluster_metric,
            cluster_radius=cluster_radius,
            # ↑↑↑
            show_debug=show_debug,
            debug_verbose=False
        )

        w = self.win
        #print(show_debug)
        if show_debug:
            self._debug_plots(signal_data = data, mark_data = np.array([]), horiz_axis_line_inds = [],
                            vert_axis_line_inds = idx, title = f"{name}  final matches (n={len(idx)})",
                            label = 0, plot_option = "matches")

        if sco.size:
            print(f"✅ {len(idx)} matches | mean r = {sco.mean():.4f}")
        else:
            print("No matches found")

        return idx, sco

    """
    Do `passes` of scan→update on a single dataset, then finalize and plot.

    Notes
    -----
    - Template is adapted after each pass from ALL accepted windows.
    - Clustering is applied during scanning to reduce near-duplicates.
    """
    def run_dataset(self, data, name,
                    amp_max, thr, rel_err, passes,
                    pos_shift, sad_thresh, min_std,
                    enforce_low_amp=True,
                    low_amp_quantile=0.10,
                    low_amp_cover=0.60,
                    # :
                    cluster_metric="sad",
                    cluster_radius=None,
                    show_debug=True, debug_verbose=False):

        for p in range(passes):
            if debug_verbose:
                print(f"   pass {p+1}/{passes}")
            idx, _ = self.scan_matches(
                data, thr, amp_max, rel_err,
                pos_shift=pos_shift,
                sad_thresh=sad_thresh,
                min_std=min_std,
                enforce_low_amp=enforce_low_amp,
                low_amp_quantile=low_amp_quantile,
                low_amp_cover=low_amp_cover,
                # ↓↓↓
                cluster_metric=cluster_metric,
                cluster_radius=cluster_radius,
                # ↑↑↑
                show_debug=show_debug,
                debug_verbose=debug_verbose
            )
            self.update_template(data, idx, top_n=20, show_debug=show_debug)
        
        # Final pass of the refined template
        final_marked_indices = self.final_template_match_and_plot(
            name, data, thr, amp_max,
            pos_shift, sad_thresh, min_std, rel_err,
            enforce_low_amp, low_amp_quantile, low_amp_cover,
            # ↓↓↓
            cluster_metric=cluster_metric,
            cluster_radius=cluster_radius,
            # ↑↑↑
            show_debug=show_debug
        )

        # Reset back to basic shape template
        self.reset()

        return final_marked_indices
