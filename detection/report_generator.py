"""
PDF Report Generation Module
Generates downloadable PDF reports for melasma detection results.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from django.conf import settings
import os
from datetime import datetime


def generate_pdf_report(report, user, prediction_result=None):
    """
    Generate PDF report for melasma detection.
    
    Args:
        report: MelasmaReport instance
        user: User instance
    
    Returns:
        File path to saved PDF
    """
    # Get user profile
    try:
        profile = user.profile
        name = profile.name
        gender = profile.gender
        # Get gender choices from the model
        from detection.models import UserProfile
        gender_display = dict(UserProfile.GENDER_CHOICES).get(gender, gender)
    except Exception:
        name = user.get_full_name() or user.username
        gender_display = 'N/A'
    
    # Create PDF buffer
    buffer = BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for PDF elements
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6B46C1'),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#4C1D95'),
        spaceAfter=12,
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
    )
    
    # Build PDF contents
    elements.append(Paragraph("MelaScan Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Patient Information (Name, Gender, Date)
    elements.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Name:', name],
        ['Gender:', gender_display],
        ['Date:', report.date.strftime('%B %d, %Y')],
    ]
    patient_table = Table(patient_data, colWidths=[2 * inch, 4 * inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.15 * inch))

    # Detection result and confidence
    elements.append(Paragraph("Detection Result", heading_style))
    result_text = report.result
    confidence_text = ''
    if prediction_result and prediction_result.get('confidence') is not None:
        try:
            confidence_text = f" (Confidence: {prediction_result['confidence']:.2f}%)"
        except Exception:
            confidence_text = f" (Confidence: {prediction_result.get('confidence')})"

    result_style = ParagraphStyle(
        'Result',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.black,
        spaceAfter=12,
    )
    elements.append(Paragraph(f"<b>{result_text}{confidence_text}</b>", result_style))
    elements.append(Spacer(1, 0.1 * inch))

    # Images: uploaded image and segmentation overlay (if available)
    elements.append(Paragraph("Images", heading_style))
    try:
        uploaded_rel = report.uploaded_image.name
        uploaded_path = os.path.join(settings.MEDIA_ROOT, uploaded_rel)
        if os.path.exists(uploaded_path):
            img = RLImage(uploaded_path, width=4 * inch, height=4 * inch)
            elements.append(Paragraph("Original Image", normal_style))
            elements.append(img)
            elements.append(Spacer(1, 0.15 * inch))
    except Exception:
        pass

    if prediction_result and prediction_result.get('overlay_path'):
        overlay = prediction_result.get('overlay_path')
        if not os.path.isabs(overlay):
            overlay_path = os.path.join(settings.MEDIA_ROOT, overlay)
        else:
            overlay_path = overlay

        if os.path.exists(overlay_path):
            try:
                img2 = RLImage(overlay_path, width=4 * inch, height=4 * inch)
                elements.append(Paragraph("Segmentation Mask", normal_style))
                elements.append(img2)
                elements.append(Spacer(1, 0.15 * inch))
            except Exception:
                pass

    # Model information
    elements.append(Paragraph("Model Information", heading_style))
    model_data = [['Model Used:', report.model_used]]
    model_table = Table(model_data, colWidths=[2 * inch, 4 * inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EDE9FE')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(model_table)

    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("© MelaScan WebApp — Generated by MelaScan", footer_style))

    # Build PDF
    doc.build(elements)

    # Save PDF to file
    pdf_filename = f'melascan_report_{report.id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    pdf_path = os.path.join(settings.MEDIA_ROOT, 'reports', datetime.now().strftime('%Y/%m/%d'))
    os.makedirs(pdf_path, exist_ok=True)
    pdf_full_path = os.path.join(pdf_path, pdf_filename)

    # Write PDF to file
    with open(pdf_full_path, 'wb') as f:
        f.write(buffer.getvalue())

    buffer.close()

    # Return relative path for FileField
    return f'reports/{datetime.now().strftime("%Y/%m/%d")}/{pdf_filename}'

