"""
PDF Report Generation Module
Generates downloadable PDF reports for melasma detection results.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from django.conf import settings
import os
from datetime import datetime


def generate_pdf_report(report, user):
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
    except:
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
    
    # Title
    elements.append(Paragraph("MelaScan Report", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Report Information
    elements.append(Paragraph("Patient Information", heading_style))
    
    # Patient data table
    patient_data = [
        ['Name:', name],
        ['Gender:', gender_display],
        ['Date:', report.date.strftime('%B %d, %Y')],
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Result section
    elements.append(Paragraph("Detection Result", heading_style))
    
    # Result with color coding
    result_color = colors.HexColor('#DC2626') if 'Melasma Detected' in report.result else colors.HexColor('#059669')
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=result_color,
        spaceAfter=12,
    )
    elements.append(Paragraph(f"<b>{report.result}</b>", result_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Model Information
    elements.append(Paragraph("Model Information", heading_style))
    
    model_data = [
        ['Best Model Used:', report.model_used],
        ['Accuracy:', f"{report.accuracy:.1%}"],
        ['Precision:', f"{report.precision:.1%}"],
        ['Recall:', f"{report.recall:.1%}"],
        ['F1 Score:', f"{report.f1_score:.1%}"],
    ]
    
    model_table = Table(model_data, colWidths=[2*inch, 4*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EDE9FE')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    elements.append(model_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("© MelaScan WebApp — Developed by Team M — 2025", footer_style))
    
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

