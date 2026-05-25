import numpy as np
import time
import cv2

class Plotter:
    def __init__(self, plot_width, plot_height, sample_buffer = None, scale_value = 1):
        self.scale_value = scale_value
        self.width = plot_width
        self.height = plot_height
        self.color = (0, 255, 0)
        self.plot_canvas = np.ones((self.height, self.width, 3))*255
        self.ltime = 0
        self.plots = {}
        self.plot_t_last = {}
        self.margin_l = 50
        self.margin_r = 50
        self.margin_u = 50
        self.margin_d = 50
        self.sample_buffer = self.width if sample_buffer is None else sample_buffer

    def plot(self, val, label = "plot", t1 = 1, t2 = 1):
        self.t1 = t1
        self.t2 = t2
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0

        self.plots[label].append(int(val * self.scale_value) / self.scale_value)
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            self.show_plot(label)

    def show_plot(self, label):
        self.plot_canvas = np.zeros((self.height, self.width, 3))

        scale_h = 8*self.scale_value*(self.height - self.margin_d - self.margin_u) / self.height
        for j, i in enumerate(np.linspace(0, self.sample_buffer-2, self.width - self.margin_l - self.margin_r)):
            i = int(i)
            color = (0, 255, 0)
            cv2.line(self.plot_canvas, (j+self.margin_l, int((self.height - self.margin_d - self.margin_u) + self.margin_u - self.plots[label][i] * scale_h)), (j+self.margin_l, int((self.height - self.margin_d - self.margin_u) + self.margin_u - self.plots[label][i+1] * scale_h)), color, 1)

        cv2.rectangle(self.plot_canvas, (self.margin_l, self.margin_u), (self.width - self.margin_r, self.height - self.margin_d), (255, 255, 255), 1)

        # Draw grid lines.
        cv2.line(self.plot_canvas, (self.margin_l, int((self.height-self.margin_d-self.margin_u)/4)+self.margin_u ),
                                   (self.width-self.margin_r, int((self.height-self.margin_d-self.margin_u)/4)+self.margin_u), (1,1,1), 1)
        cv2.line(self.plot_canvas, (self.margin_l, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u ),
                                   (self.width-self.margin_r, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u), (1,1,1), 1)
        cv2.line(self.plot_canvas, (self.margin_l, int((self.height-self.margin_d-self.margin_u)*3/4)+self.margin_u ),
                                   (self.width-self.margin_r, int((self.height-self.margin_d-self.margin_u)*3/4)+self.margin_u), (1,1,1), 1, )

        # Add y-axis gridline values.
        fontType = cv2.FONT_HERSHEY_TRIPLEX
        font_adjust = 5
        cv2.putText(self.plot_canvas,f"{0.50}", (int(font_adjust), int(0)+self.margin_u + font_adjust), fontType, 0.5, (255,255,255))
        cv2.putText(self.plot_canvas,f"{0.25}", (int(font_adjust), int((self.height-self.margin_d-self.margin_u)*1/2)+self.margin_u + font_adjust), fontType, 0.5, (255,255,255))
        cv2.putText(self.plot_canvas,f"{0.0}", (int(font_adjust+21), int((self.height-self.margin_d-self.margin_u)*4/4)+self.margin_u + font_adjust), fontType, 0.5, (255,255,255))

        color = (0,255,255)
        cv2.putText(self.plot_canvas,f" {label} : {self.plots[label][-1]}", (int(self.width/2 - 50), self.height-20), fontType, 0.6, color)

        self.plot_t_last[label] = time.time()
        cv2.imshow(label, self.plot_canvas)
        cv2.waitKey(1)