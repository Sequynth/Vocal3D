from typing import List, Tuple

import numpy as np
import zoomableVideo
from PyQt5.QtCore import QLineF, QPointF
from PyQt5.QtGui import QBrush, QColor, QImage, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem,
                             QGraphicsPixmapItem, QGraphicsPolygonItem, QMenu)


class FeatureViewer(zoomableVideo.ZoomableVideo):
    def __init__(
        self,
        video: List[QImage],
        points: np.array = None,
        vocalfold_segmentations: List[QImage] = None,
        glottal_segmentations: List[QImage] = None,
        glottal_outlines: List[np.array] = None,
        glottal_midlines: List[Tuple[np.array, np.array]] = None,
        parent=None,
    ):
        super(FeatureViewer, self).__init__(video, parent)

        if video is not None:
            self._image_width = video[0].width()
            self._image_height = video[0].height()


        self.feature_dicts = {}

        self._point_positions = points
        self._point_items: List[QGraphicsEllipseItem] = []
        self._pen_point = QPen(QColor(128, 255, 128, 255))
        self._brush_point = QBrush(QColor(128, 255, 128, 128))
        self._point_size: int = 3
        self._show_points: bool = True
        self._opacity_points: float = 1.0
        self.feature_dicts["P"] = {"show": self.show_point, "opacity": self.set_point_opacity}

        self._optimized_points = points
        self._optimized_point_items: List[QGraphicsEllipseItem] = []
        self._pen_op = QPen(QColor(128, 128, 255, 255))
        self._brush_op = QBrush(QColor(128, 128, 255, 128))
        self._show_op: bool = True
        self._opacity_op: float = 1.0
        self.feature_dicts["OP"] = {"show": self.show_optimized_points, "opacity": self.set_optimized_point_opacity}

        self._glottal_midlines = glottal_midlines
        self._item_gm: QGraphicsLineItem = None
        self._pen_gm = QPen(QColor(0, 255, 255, 255))
        self._brush_gm = QBrush(QColor(0, 255, 255, 128))
        self._show_gm: bool = True
        self._opacity_gm: float = 1.0
        self.feature_dicts["GM"] = {"show": self.show_glottal_midline, "opacity": self.set_glottal_midline_opacity}

        self._glottal_outlines: List[np.array] = glottal_outlines
        self._item_go: QGraphicsPolygonItem = None
        self._pen_go = QPen(QColor(0, 0, 255, 0))
        self._brush_go = QBrush(QColor(0, 0, 255, 255))
        self._show_go: bool = True
        self._opacity_go: float = 1.0
        self.feature_dicts["GO"] = {"show": self.show_glottal_outline, "opacity": self.set_glottal_outline_opacity}

        self._glottis_segmentations = glottal_segmentations
        self._item_gs: QGraphicsPixmapItem = None
        self._show_gs: bool = True
        self._opacity_gs: float = 1.0
        self.feature_dicts["GS"] = {"show": self.show_glottal_segmentation, "opacity": self.set_glottal_segmentation_opacity}

        self._vocalfold_segmentations = vocalfold_segmentations
        self._item_vfs: QGraphicsPixmapItem = None
        self._show_vfs: bool = True
        self._opacity_vfs: float = 1.0
        self.feature_dicts["VS"] = {"show": self.show_vocalfold_segmentation, "opacity": self.set_vocalfold_opacity}


    def add_points(self, points: np.array) -> None:
        # Points should be in FRAMES x NUM_POINTS x 2 in [Y, X]
        self._point_positions = points

    def add_optimized_points(self, points: np.array) -> None:
        # Points should be in FRAMES x NUM_POINTS x 2 in [Y, X]
        self._optimized_points = points

    def add_vocalfold_segmentations(self, vfs: List[QImage]):
        self._vocalfold_segmentations = vfs

    def add_glottal_segmentations(self, gs: List[QImage]):
        self._glottis_segmentations = gs

    def add_glottal_midlines(self, gms: np.array):
        self._glottal_midlines = gms

    def add_glottal_outlines(self, gos: List[np.array]):
        self._glottal_outlines = gos
    
    def show_optimized_points(self, show: bool):
        self._show_op = show
    
    def show_point(self, show: bool):
        self._show_points = show

    def show_vocalfold_segmentation(self, show: bool):
        self._show_vfs = show

    def show_glottal_outline(self, show: bool):
        self._show_go = show

    def show_glottal_segmentation(self, show: bool):
        self._show_gs = show

    def show_glottal_midline(self, show: bool):
        self._show_gm = show

    def set_point_opacity(self, opacity: float):
        self._opacity_points = opacity / 100

    def set_optimized_point_opacity(self, opacity: float):
        self._opacity_op = opacity / 100

    def set_vocalfold_opacity(self, opacity: float):
        self._opacity_vfs = opacity / 100

    def set_glottal_outline_opacity(self, opacity: float):
        self._opacity_go = opacity / 100

    def set_glottal_segmentation_opacity(self, opacity: float):
        self._opacity_gs = opacity / 100

    def set_glottal_midline_opacity(self, opacity: float):
        self._opacity_gm = opacity / 100

    def contextMenuEvent(self, event) -> None:
        """
        Opens a context menu with options for zooming in and out.

        :param event: The QContextMenuEvent containing information about the context menu event.
        :type event: QContextMenuEvent
        """
        menu = QMenu()
        menu.addAction("Zoom in               MouseWheel Up", self.zoomIn)
        menu.addAction("Zoom out              MouseWheel Down", self.zoomOut)
        menu.addAction("Reset View", self.zoomReset)
        menu.addAction("Fit View", self.fit_view)
        menu.exec_(event.globalPos())

    def redraw(self) -> None:
        if self.images is not None:
            self.set_image(self.images[self._current_frame])

        if self._show_points:
            self.draw_points()

        if self._show_op:
            self.draw_optimized_points()
        
        if self._show_vfs:
            self.draw_vocalfold_segmentation()
        
        if self._show_gs:
            self.draw_glottal_segmentation()
        
        if self._show_go:
            self.draw_glottal_outline()

        if self._show_gm:
            self.draw_glottal_midline()


    def draw_glottal_midline(self):
        if self._glottal_midlines is None:
            return
        
        self.scene().removeItem(self._item_gm)
        x1, y1 = self._glottal_midlines[self._current_frame][0].tolist()
        x2, y2 = self._glottal_midlines[self._current_frame][1].tolist()

        self._item_gm = self.scene().addLine(x1, y1, x2, y2, self._pen_gm)
        self._item_gm.setOpacity(self._opacity_gm)

    def draw_vocalfold_segmentation(self):
        if self._vocalfold_segmentations is None:
            return

        self.scene().removeItem(self._item_vfs)
        self._item_vfs = self.scene().addPixmap(QPixmap(self._vocalfold_segmentations[self._current_frame]))
        self._item_vfs.setOpacity(self._opacity_vfs)
    
    def draw_glottal_segmentation(self):
        if self._glottis_segmentations is None:
            return
        

        self.scene().removeItem(self._item_gs)
        self._item_gs = self.scene().addPixmap(QPixmap(self._glottis_segmentations[self._current_frame]))
        self._item_gs.setOpacity(self._opacity_gs)
    
    def draw_glottal_outline(self):
        if self._glottal_outlines is None:
            return
        
        self.scene().removeItem(self._item_go)
        polygon_points: List[QPointF] = [QPointF(x[1], x[0]) for x in self._glottal_outlines[self._current_frame]]
        self._item_go = self.scene().addPolygon(QPolygonF(polygon_points), self._pen_go, self._brush_go)
        self._item_go.setOpacity(self._opacity_go)
        

    def draw_points(self):
        if self._point_positions is None:
            return

        for point_item in self._point_items:
            self.scene().removeItem(point_item)
        self._point_items = []

        # Get current frame indices:
        points = self._point_positions[self._current_frame]
        for point in points:
            if np.isnan(point).any():
                continue

            ellipse_item = self.scene().addEllipse(
                point[1] - self._point_size / 2,
                point[0] - self._point_size / 2,
                self._point_size,
                self._point_size,
                self._pen_point,
                self._brush_point,
            )
            self._point_items.append(ellipse_item)
        

    def draw_optimized_points(self):
        if self._optimized_points is None:
            return

        for point_item in self._optimized_point_items:
            self.scene().removeItem(point_item)
        self._optimized_point_items = []

        # Get current frame indices:
        points = self._optimized_points[self._current_frame]
        for point in points:
            if np.isnan(point).any():
                continue

            ellipse_item = self.scene().addEllipse(
                point[1] - self._point_size / 2,
                point[0] - self._point_size / 2,
                self._point_size,
                self._point_size,
                self._pen_op,
                self._brush_op,
            )
            self._optimized_point_items.append(ellipse_item)

    def keyPressEvent(self, event) -> None:
        self.change_frame(self._current_frame + 1)
        self.fit_view()
