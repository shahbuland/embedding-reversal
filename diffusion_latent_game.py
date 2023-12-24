import pygame
import numpy as np
import cv2

class CVBlock:
    def __init__(self, name : str, width=512, height=512):
        self.width = width
        self.height = height
        self.name = name
        self.current_image = np.zeros((height, width, 3), np.uint8)
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.imshow(self.name, self.current_image)

    def destroy(self):
        cv2.destroyAllWindows()

class CVImageWindow(CVBlock):
    def update(self, pil_image):
        try:
            pil_image = pil_image.resize((self.width, self.height))
            self.current_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, self.current_image)
        except:
            pass

class PointRendererImages:
    def __init__(self, width=1500, height=1000, point_size = 4, callback = None):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.width = width
        self.height = height
        self.point_size = point_size
        
        self.screen_margin = 0.1

        self.blocks = {
            'image' : CVImageWindow("PokemonImage")
        }

    def map_to_screen(self, points):
        x = points[:, 0]
        y = points[:, 1]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x = (x - x_min) / (x_max - x_min) * (self.width - 2 * self.screen_margin * self.width) + self.screen_margin * self.width
        y = self.height - ((y - y_min) / (y_max - y_min) * (self.height - 2 * self.screen_margin * self.height) + self.screen_margin * self.height)
        
        return np.column_stack((x, y))

    def quit(self):
        for key in self.blocks:
            self.blocks[key].destroy()
        
    def run(self, points, data = None, colors=None):
        points = self.map_to_screen(points)
        if colors is None:
            colors = [(255, 255, 255)] * len(points)
        current_image = None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    for i, point in enumerate(points):
                        if np.linalg.norm(np.array(point) - np.array([x, y])) < 2:
                            if current_image is not None:
                                current_image.close()
                            current_image = data[i]
                            self.blocks['image'].update(current_image)
                        else:
                            current_image = self.callback(x, y)
                            
            self.screen.fill((0, 0, 0))
            for point, color in zip(points, colors):
                pygame.draw.circle(self.screen, color, point, self.point_size)
            pygame.display.flip()
            self.clock.tick(60)