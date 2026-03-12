from datetime import datetime
from sqlalchemy import DateTime, String, Text, text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from geoalchemy2 import Geography


class Base(DeclarativeBase):
    pass


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[str] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()::text"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"))
    encrypted_session_key: Mapped[str | None] = mapped_column(Text, default=None)
    video_path: Mapped[str | None] = mapped_column(String(512), default=None)
    description_full: Mapped[str | None] = mapped_column(Text, default=None)
    description_short: Mapped[str | None] = mapped_column(Text, default=None)
    location = mapped_column(Geography(geometry_type="POINT", srid=4326), nullable=True)
    accuracy: Mapped[float | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(Text, default="pending")
    frames: Mapped[list["ReportedFrame"]] = relationship(back_populates="report")
    user_id: Mapped[str | None] = mapped_column(Text, default=None)

class ReportedFrame(Base):
    __tablename__ = "reported_frame"

    reported_frame_id: Mapped[str] = mapped_column(primary_key=True, server_default=text("gen_random_uuid()::text"))
    report_id: Mapped[str] = mapped_column(ForeignKey("reports.report_id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    report: Mapped["Report"] = relationship(back_populates="frames")

