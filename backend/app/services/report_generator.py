import os
import base64
import tempfile
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor


class InsuranceReportGenerator:
    
    # Colors
    PRIMARY = HexColor("#1e3a5f")
    SECONDARY = HexColor("#4a6fa5")
    ACCENT = HexColor("#2e7d32")
    TEXT_DARK = HexColor("#212121")
    TEXT_LIGHT = HexColor("#757575")
    BORDER = HexColor("#e0e0e0")
    BG_LIGHT = HexColor("#f5f5f5")
    
    def __init__(self, output_dir: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_out = os.path.join(base_dir, "../../../outputs/reports")
        self.output_dir = output_dir or default_out
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_image_reader(self, roi):
        fp = roi.get("file_path")
        if fp and os.path.exists(fp):
            return ImageReader(fp), None

        b64 = roi.get("base64")
        if b64:
            b = base64.b64decode(b64)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.write(b)
            tmp.flush()
            tmp.close()
            return ImageReader(tmp.name), tmp.name

        return None, None

    def _draw_header(self, c, width, height, claim_id):
        # Header background
        c.setFillColor(self.PRIMARY)
        c.rect(0, height - 35 * mm, width, 35 * mm, fill=True, stroke=False)
        
        # Title
        c.setFillColor(HexColor("#ffffff"))
        c.setFont("Helvetica-Bold", 20)
        c.drawString(20 * mm, height - 18 * mm, "Vehicle Damage Assessment Report")
        
        # Subtitle
        c.setFont("Helvetica", 10)
        c.drawString(20 * mm, height - 26 * mm, f"Claim Reference: {claim_id}")
        c.drawRightString(width - 20 * mm, height - 26 * mm, 
                         f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}")

    def _draw_summary_box(self, c, y, width, stage5_output):
        box_height = 45 * mm
        margin = 20 * mm
        box_width = width - (2 * margin)
        
        # Box background
        c.setFillColor(self.BG_LIGHT)
        c.roundRect(margin, y - box_height, box_width, box_height, 3 * mm, fill=True, stroke=False)
        
        # Border
        c.setStrokeColor(self.BORDER)
        c.setLineWidth(0.5)
        c.roundRect(margin, y - box_height, box_width, box_height, 3 * mm, fill=False, stroke=True)
        
        # Title
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin + 5 * mm, y - 8 * mm, "Cost Estimate Summary")
        
        # Total cost (prominent)
        total_cost = stage5_output.get("total_estimated_cost", 0)
        c.setFillColor(self.ACCENT)
        c.setFont("Helvetica-Bold", 24)
        c.drawString(margin + 5 * mm, y - 22 * mm, f"₹{total_cost:,.0f}")
        
        c.setFillColor(self.TEXT_LIGHT)
        c.setFont("Helvetica", 9)
        c.drawString(margin + 5 * mm, y - 28 * mm, "Estimated Repair Cost")
        
        # Stats on the right
        stats_x = margin + 90 * mm
        c.setFillColor(self.TEXT_DARK)
        c.setFont("Helvetica", 10)
        
        valid = stage5_output.get("valid_regions", 0)
        ignored = stage5_output.get("ignored_regions", 0)
        
        c.drawString(stats_x, y - 15 * mm, f"Damage Areas Identified: {valid}")
        c.drawString(stats_x, y - 23 * mm, f"Areas Below Threshold: {ignored}")
        c.drawString(stats_x, y - 31 * mm, f"Assessment Date: {datetime.now().strftime('%Y-%m-%d')}")
        
        return y - box_height - 10 * mm

    def _draw_table_header(self, c, y, margin):
        c.setFillColor(self.SECONDARY)
        c.rect(margin, y - 8 * mm, 170 * mm, 8 * mm, fill=True, stroke=False)
        
        c.setFillColor(HexColor("#ffffff"))
        c.setFont("Helvetica-Bold", 9)
        c.drawString(margin + 3 * mm, y - 6 * mm, "Part")
        c.drawString(margin + 45 * mm, y - 6 * mm, "Severity")
        c.drawString(margin + 75 * mm, y - 6 * mm, "Damage Type")
        c.drawString(margin + 115 * mm, y - 6 * mm, "Coverage")
        c.drawString(margin + 140 * mm, y - 6 * mm, "Est. Cost (INR)")
        
        return y - 8 * mm

    def _draw_table_row(self, c, y, margin, row, is_alt):
        row_height = 7 * mm
        
        if is_alt:
            c.setFillColor(self.BG_LIGHT)
            c.rect(margin, y - row_height, 170 * mm, row_height, fill=True, stroke=False)
        
        c.setFillColor(self.TEXT_DARK)
        c.setFont("Helvetica", 9)
        
        part = str(row.get("part", "Unknown")).title()
        severity = str(row.get("severity", "-")).title()
        damage_type = str(row.get("damage_type", "-")).replace("_", " ").title()
        coverage = f"{row.get('coverage_percent', 0):.1f}%"
        cost = f"₹{row.get('final_cost', 0):,.0f}"
        
        c.drawString(margin + 3 * mm, y - 5 * mm, part[:20])
        c.drawString(margin + 45 * mm, y - 5 * mm, severity)
        c.drawString(margin + 75 * mm, y - 5 * mm, damage_type)
        c.drawString(margin + 115 * mm, y - 5 * mm, coverage)
        c.drawString(margin + 140 * mm, y - 5 * mm, cost)
        
        return y - row_height

    def _draw_footer(self, c, width, page_num):
        c.setStrokeColor(self.BORDER)
        c.setLineWidth(0.5)
        c.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
        
        c.setFillColor(self.TEXT_LIGHT)
        c.setFont("Helvetica", 8)
        c.drawString(20 * mm, 10 * mm, "This is an automated assessment. Final costs may vary based on inspection.")
        c.drawRightString(width - 20 * mm, 10 * mm, f"Page {page_num}")

    def generate(self, stage4_output: dict, stage5_output: dict, meta: dict = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        claim_id = meta.get("claim_id") if meta and meta.get("claim_id") else f"CLM-{ts}"
        filename = f"insurance_report_{claim_id}_{ts}.pdf"
        out_path = os.path.join(self.output_dir, filename)

        c = canvas.Canvas(out_path, pagesize=A4)
        width, height = A4
        margin = 20 * mm
        page_num = 1

        # Header
        self._draw_header(c, width, height, claim_id)
        y = height - 45 * mm

        # Summary box
        y = self._draw_summary_box(c, y, width, stage5_output)
        y -= 5 * mm

        # Damage details section
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Damage Breakdown")
        y -= 10 * mm

        # Table
        y = self._draw_table_header(c, y, margin)
        
        region_list = stage5_output.get("details", [])
        tmp_files = []

        for idx, row in enumerate(region_list):
            if y < 30 * mm:
                self._draw_footer(c, width, page_num)
                c.showPage()
                page_num += 1
                y = height - 25 * mm
                y = self._draw_table_header(c, y, margin)
            
            y = self._draw_table_row(c, y, margin, row, idx % 2 == 1)

        # Table bottom border
        c.setStrokeColor(self.BORDER)
        c.setLineWidth(0.5)
        c.line(margin, y, margin + 170 * mm, y)
        y -= 15 * mm

        # ROI Images section
        roi_images = stage4_output.get("roi_images", [])
        if roi_images:
            if y < 80 * mm:
                self._draw_footer(c, width, page_num)
                c.showPage()
                page_num += 1
                y = height - 25 * mm

            c.setFillColor(self.PRIMARY)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, "Damage Evidence")
            y -= 10 * mm

            img_x = margin
            thumb_w = 55 * mm
            thumb_h = 42 * mm
            imgs_per_row = 3
            
            for idx, roi in enumerate(roi_images):
                if idx > 0 and idx % imgs_per_row == 0:
                    y -= (thumb_h + 8 * mm)
                    img_x = margin
                
                if y - thumb_h < 25 * mm:
                    self._draw_footer(c, width, page_num)
                    c.showPage()
                    page_num += 1
                    y = height - 25 * mm
                    img_x = margin

                img_reader, tmpf = self._get_image_reader(roi)
                if img_reader:
                    # Image border
                    c.setStrokeColor(self.BORDER)
                    c.setLineWidth(1)
                    c.rect(img_x - 1 * mm, y - thumb_h - 1 * mm, 
                           thumb_w + 2 * mm, thumb_h + 2 * mm, fill=False, stroke=True)
                    
                    c.drawImage(img_reader, img_x, y - thumb_h, 
                               width=thumb_w, height=thumb_h, preserveAspectRatio=True)
                    
                    if tmpf:
                        tmp_files.append(tmpf)
                
                img_x += thumb_w + 5 * mm

        # Footer
        self._draw_footer(c, width, page_num)
        c.save()
        for f in tmp_files:
            try:
                os.remove(f)
            except Exception:
                pass

        return out_path