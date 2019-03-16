import cv2
import numpy as np

class HOG():
    
    def __init__(self, cellsize=8, blocksize=2, n_hist=9):
        
        self.cellsize = cellsize
        self.blocksize = blocksize
        self.n_hist = n_hist
        
    def fit(self, image):
        """
        Params:
            image:  {ndarray(H, W, C)}
        """
        h, w, c = image.shape

        assert h%self.cellsize==0 and w%self.cellsize==0, "image size error"

        grad_x, grad_y = self._grad(image)
        grad = np.sqrt(grad_x**2 + grad_y**2)

        mask = (grad_x!=0)
        angle = np.zeros(shape=(h, w, c))
        angle[mask] = np.arctan(grad_y[mask] / grad_x[mask]) * 180 / np.pi  # (-90, 90)
        angle = angle.astype('int') + 90

        # cv2.imshow('x', grad_x)
        # cv2.imshow('y', grad_y)
        # cv2.imshow('g', grad)
        # cv2.imshow('a', angle)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        hog_w = w // self.cellsize
        hog_h = h // self.cellsize
        histsize = 180 // self.n_hist

        hog = np.zeros(shape=(hog_h, hog_w, c, self.n_hist))
        for i in range(hog_h):
            for j in range(hog_w):

                y1, y2, x1, x2 = i*self.cellsize, (i+1)*self.cellsize, j*self.cellsize, (j+1)*self.cellsize
                gcell = grad[y1: y2, x1: x2]    # (H, W, C)
                acell = angle[y1: y2, x1: x2]   # (H, W, C)
                
                for m in range(c):

                    gcell_c = gcell[:, :, m]    # (H, W)
                    acell_c = acell[:, :, m]    # (H, W)

                    for ii in range(self.cellsize):
                        for jj in range(self.cellsize):
                            a = acell_c[ii, jj]
                            g = gcell_c[ii, jj]
                            hog[i, j, m, a // histsize] += a * g

        # for i in range(self.n_hist):
        #     cv2.imshow("", hog[:, :, :, i])
        #     cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return hog

    
    def _pad(self, image, p=1):
        """
        Params:
            image:  {ndarray(H, W, C)}
        Returns:
            pad:  {ndarray(H+2p, W+2p, C)}
        """
        h, w, c = image.shape
        pad = np.zeros(shape=(h+2*p, w+2*p, c))
        pad[p: -p, p: -p, :] = image
        return pad

    def _grad(self, image, s=2):
        """
        Params:
            image:  {ndarray(H, W, C)}
        Returns:
            grad_x: {ndarray(H-s, W-s, C)}
            grad_y: {ndarray(H-s, W-s, C)}
        """
        h, w, c = image.shape

        grad_x = np.zeros(shape=(h, w + s, c))
        grad_x[:, s//2: -s//2] = image
        grad_x = grad_x[:, :-s] - grad_x[:, s:]

        grad_y = np.zeros(shape=(h+s, w, c))
        grad_y[s//2: -s//2, :] = image
        grad_y = grad_y[:-s, :] - grad_y[s:, :]

        return grad_x, grad_y


if __name__ == "__main__":

    image = cv2.imread('./joker.jpg')
    # cv2.imshow("", image); cv2.waitKey(0)

    hog = HOG()
    hog.fit(image)