import os
import io
import re
import sqlite3
import zipfile
import json
import math
import uuid
import datetime as dt
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
import pytesseract
import fitz
from PIL import Image
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, UniqueConstraint, text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
import gradio as gr

AZUL_MARINHO = "#0A1F44"
TURQUESA = "#1CC8C8"
TURQUESA_CLARO = "#5DE0E6"
DOURADO = "#F2C14E"
DOURADO_CLARO = "#FFE49C"
BRANCO = "#FFFFFF"

LINGUAS = ["Inglês", "Espanhol"]
LINGUA_COMUM = "Comum"

DB_PATH = "olimpo.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
Base = declarative_base()
Session = sessionmaker(bind=engine, future=True)

class Aluno(Base):
    __tablename__ = "alunos"
    id = Column(Integer, primary_key=True)
    numero = Column(Integer, unique=True, index=True)
    nome = Column(String, default="")
    turno = Column(String, default="Manhã")
    turma = Column(String, default="")
    lingua_estrangeira = Column(String, default="Inglês")
    ativo = Column(Boolean, default=True)

class Prova(Base):
    __tablename__ = "provas"
    id = Column(Integer, primary_key=True)
    nome = Column(String)
    data = Column(DateTime, default=dt.datetime.utcnow)
    disciplina = Column(String, default="")
    area = Column(String, default="Humanas")
    numero_questoes = Column(Integer, default=45)
    turma = Column(String, default="")
    tem_lingua = Column(Boolean, default=False)
    questoes_lingua = Column(String, default="")
    pdf_origem = Column(String, nullable=True)

class Gabarito(Base):
    __tablename__ = "gabaritos"
    id = Column(Integer, primary_key=True)
    prova_id = Column(Integer, ForeignKey("provas.id"))
    numero_questao = Column(Integer)
    resposta = Column(String)
    lingua = Column(String, default=LINGUA_COMUM)
    origem = Column(String, default="manual")
    __table_args__ = (UniqueConstraint("prova_id", "numero_questao", "lingua", name="uq_gab_lang"),)

class RespostaAluno(Base):
    __tablename__ = "respostas"
    id = Column(Integer, primary_key=True)
    prova_id = Column(Integer, ForeignKey("provas.id"))
    aluno_id = Column(Integer, ForeignKey("alunos.id"))
    numero_questao = Column(Integer)
    resposta = Column(String, nullable=True)
    correta = Column(Boolean, default=False)
    confianca = Column(Float, default=0.0)
    revisao_manual = Column(Boolean, default=False)
    lingua_aplicada = Column(String, default=LINGUA_COMUM)

class Resultado(Base):
    __tablename__ = "resultados"
    id = Column(Integer, primary_key=True)
    aluno_id = Column(Integer, ForeignKey("alunos.id"))
    prova_id = Column(Integer, ForeignKey("provas.id"))
    acertos = Column(Integer, default=0)
    percentual = Column(Float, default=0.0)
    tri = Column(Float, nullable=True)

class Processamento(Base):
    __tablename__ = "processamentos"
    id = Column(Integer, primary_key=True)
    data = Column(DateTime, default=dt.datetime.utcnow)
    arquivo = Column(String)
    status = Column(String, default="Pendente")
    confianca = Column(Float, default=0.0)
    erros = Column(String, nullable=True)
    pagina = Column(Integer, default=1)
    tipo = Column(String, default="imagem")

class LogOperacao(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True)
    data = Column(DateTime, default=dt.datetime.utcnow)
    mensagem = Column(String)
    nivel = Column(String, default="Info")

Base.metadata.create_all(engine)

def _migrar():
    cols = [
        ("alunos", "lingua_estrangeira", "TEXT DEFAULT 'Inglês'"),
        ("provas", "tem_lingua", "INTEGER DEFAULT 0"),
        ("provas", "questoes_lingua", "TEXT DEFAULT ''"),
        ("provas", "pdf_origem", "TEXT"),
        ("gabaritos", "lingua", "TEXT DEFAULT 'Comum'"),
        ("gabaritos", "origem", "TEXT DEFAULT 'manual'"),
        ("respostas", "lingua_aplicada", "TEXT DEFAULT 'Comum'"),
        ("processamentos", "pagina", "INTEGER DEFAULT 1"),
        ("processamentos", "tipo", "TEXT DEFAULT 'imagem'")
    ]
    with engine.begin() as conn:
        for tabela, col, tipo in cols:
            try:
                conn.execute(text(f"ALTER TABLE {tabela} ADD COLUMN {col} {tipo}"))
            except Exception:
                pass

_migrar()

def log(msg, nivel="Info"):
    with Session() as s:
        s.add(LogOperacao(mensagem=msg, nivel=nivel))
        s.commit()
    print(f"[{nivel}] {msg}")

def parse_questoes_lingua(spec: str) -> set:
    if not spec:
        return set()
    resultado = set()
    for parte in spec.replace(";", ",").split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-", 1)
            try:
                ini, fim = int(a.strip()), int(b.strip())
                resultado.update(range(min(ini, fim), max(ini, fim) + 1))
            except ValueError:
                pass
        elif parte.isdigit():
            resultado.add(int(parte))
    return resultado

class PdfService:
    DPI_PADRAO = 300
    PALAVRAS_ING = {"ingles", "inglês", "english", "ing", "en"}
    PALAVRAS_ESP = {"espanhol", "español", "spanish", "esp", "es"}

    @classmethod
    def pdf_para_imagens(cls, pdf_bytes: bytes, dpi: int = DPI_PADRAO) -> List[np.ndarray]:
        imgs = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8)
                arr = arr.reshape(pix.height, pix.width, pix.n)
                if pix.n == 3:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif pix.n == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                imgs.append(img)
        return imgs

    @classmethod
    def extrair_texto(cls, pdf_bytes: bytes) -> str:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)

    @classmethod
    def contar_paginas(cls, pdf_bytes: bytes) -> int:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return doc.page_count

    @classmethod
    def detectar_lingua_no_texto(cls, texto: str) -> str:
        t = texto.lower()
        tem_ing = any(p in t for p in cls.PALAVRAS_ING)
        tem_esp = any(p in t for p in cls.PALAVRAS_ESP)
        if tem_ing and not tem_esp:
            return "Inglês"
        if tem_esp and not tem_ing:
            return "Espanhol"
        return LINGUA_COMUM

    @classmethod
    def parse_gabarito_texto(cls, texto: str) -> Dict[int, str]:
        if not texto:
            return {}
        t = texto.replace("–", "-").replace("—", "-")
        respostas: Dict[int, str] = {}
        for m in re.finditer(r"(?:Q(?:UESTÃO)?\s*)?(\d{1,3})\s*[\-\.\)\:]\s*([A-Ea-e])\b", t, re.IGNORECASE):
            q, a = int(m.group(1)), m.group(2).upper()
            if 1 <= q <= 200:
                respostas.setdefault(q, a)
        for m in re.finditer(r"(?:^|\n)\s*(\d{1,3})\s*\n\s*([A-Ea-e])\b", t):
            q, a = int(m.group(1)), m.group(2).upper()
            if 1 <= q <= 200:
                respostas.setdefault(q, a)
        inline = re.findall(r"(\d{1,3})\s*([A-Ea-e])(?=\s|\d|$)", t)
        if len(inline) >= 5:
            for qs, a in inline:
                q = int(qs)
                if 1 <= q <= 200:
                    respostas.setdefault(q, a.upper())
        if not respostas:
            letras = re.findall(r"\b([A-Ea-e])\b", t)
            if len(letras) >= 5:
                for i, a in enumerate(letras, start=1):
                    respostas[i] = a.upper()
        return dict(sorted(respostas.items()))

    @classmethod
    def parse_gabarito_pdf(cls, pdf_bytes: bytes):
        n_pag = cls.contar_paginas(pdf_bytes)
        texto_total = cls.extrair_texto(pdf_bytes)
        usou_ocr = False
        if len(texto_total.strip()) < 20:
            usou_ocr = True
            imgs = cls.pdf_para_imagens(pdf_bytes, dpi=300)
            partes = []
            for pg in imgs:
                bin_ = ImageProcessingService.preprocessar(pg)
                partes.append(pytesseract.image_to_string(bin_, lang="por+eng+spa"))
            texto_total = "\n".join(partes)
        blocos = cls._separar_blocos_lingua(texto_total)
        por_lingua: Dict[str, Dict[int, str]] = {}
        if blocos:
            for lingua, trecho in blocos.items():
                por_lingua[lingua] = cls.parse_gabarito_texto(trecho)
        else:
            lingua_global = cls.detectar_lingua_no_texto(texto_total)
            por_lingua[lingua_global] = cls.parse_gabarito_texto(texto_total)
        return por_lingua, texto_total, n_pag, usou_ocr

    @classmethod
    def _separar_blocos_lingua(cls, texto: str) -> Dict[str, str]:
        padrao = re.compile(
            r"(?im)^.*?(INGL[EE]S|ESPANHOL|ESP[AA]OL|L[II]NGUA\s+ESTRANGEIRA.*?INGL[ÊE]S|L[II]NGUA\s+ESTRANGEIRA.*?ESPANHOL).*$"
        )
        matches = list(padrao.finditer(texto))
        if not matches:
            return {}
        blocos: Dict[str, str] = {}
        for i, m in enumerate(matches):
            titulo = m.group(1).lower()
            lingua = "Inglês" if "ingl" in titulo else "Espanhol"
            ini = m.end()
            fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            blocos[lingua] = texto[ini:fim]
        if matches[0].start() > 0:
            blocos[LINGUA_COMUM] = texto[:matches[0].start()]
        return blocos

class ImageProcessingService:
    @staticmethod
    def qualidade(img: np.ndarray) -> float:
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        nitidez = min(lap / 300.0, 1.0)
        tamanho = min(gray.size / (800 * 600), 1.0)
        return float(0.7 * nitidez + 0.3 * tamanho)

    @staticmethod
    def preprocessar(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_ = cv2.medianBlur(bin_, 3)
        return bin_

class OcrService:
    DIGITS_CFG = "--psm 7 -c tessedit_char_whitelist=0123456789"

    @classmethod
    def ler_numero_aluno(cls, img, roi=None):
        if roi:
            x, y, w, h = roi
            img = img[y:y + h, x:x + w]
        bin_ = ImageProcessingService.preprocessar(img)
        data = pytesseract.image_to_data(bin_, config=cls.DIGITS_CFG, output_type=pytesseract.Output.DICT)
        candidatos = []
        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if not txt.isdigit():
                continue
            conf = float(data["conf"][i]) / 100.0
            val = int(txt)
            if 1 <= val <= 60:
                candidatos.append((val, conf))
            else:
                fix = cls._tentar_corrigir(txt)
                if fix:
                    candidatos.append((fix, conf * 0.6))
        if not candidatos:
            return (None, 0.0, [])
        agregado = {}
        for v, c in candidatos:
            agregado[v] = max(agregado.get(v, 0), c)
        ordenado = sorted(agregado.items(), key=lambda x: -x[1])
        return (ordenado[0][0], ordenado[0][1], ordenado)

    @staticmethod
    def _tentar_corrigir(txt):
        mapa = {"8": "3", "6": "5", "9": "4", "0": "6", "7": "1"}
        novo = "".join(mapa.get(d, d) for d in txt)
        if novo.isdigit():
            v = int(novo)
            if 1 <= v <= 60:
                return v
        return None

    @staticmethod
    def ler_marcacoes(img, total_questoes, alternativas=("A", "B", "C", "D", "E")):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, bin_ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bolhas = []
        for c in cnts:
            (x, y), r = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            if 8 < r < 40 and area > 50:
                per = cv2.arcLength(c, True)
                if per == 0:
                    continue
                circ = 4 * math.pi * area / (per * per)
                if circ > 0.55:
                    mask = np.zeros_like(bin_)
                    cv2.circle(mask, (int(x), int(y)), int(r * 0.7), 255, -1)
                    dens = cv2.mean(bin_, mask=mask)[0] / 255.0
                    bolhas.append({"x": float(x), "y": float(y), "r": float(r), "dens": float(dens)})
        if not bolhas:
            return []
        bolhas.sort(key=lambda b: (round(b["y"] / 30), b["x"]))
        linhas = []
        for b in bolhas:
            if linhas and abs(b["y"] - linhas[-1][-1]["y"]) < 20:
                linhas[-1].append(b)
            else:
                linhas.append([b])
        resultados = []
        for i, linha in enumerate(linhas[:total_questoes], start=1):
            linha.sort(key=lambda b: b["x"])
            linha = linha[:len(alternativas)]
            densidades = [b["dens"] for b in linha]
            if not densidades:
                continue
            idx_max = int(np.argmax(densidades))
            d_max = densidades[idx_max]
            ds = sorted(densidades, reverse=True)
            segunda = ds[1] if len(ds) > 1 else 0
            if d_max < 0.10:
                alt, conf, amb = None, 0.9, False
            elif segunda > 0.5 * d_max and d_max > 0.25:
                alt, conf, amb = alternativas[idx_max], d_max, True
            else:
                alt = alternativas[idx_max]
                conf = min(d_max * 1.5, 0.99)
                amb = False
            resultados.append({"questao": i, "alternativa": alt, "confianca": conf, "ambigua": amb})
        return resultados

class TriRasch:
    MIN_ALUNOS = 30
    MIN_ITENS = 10

    @classmethod
    def estimar(cls, respostas, n_itens):
        por_aluno = {}
        for r in respostas:
            por_aluno.setdefault(r["aluno_id"], []).append(r["correta"])
        if len(por_aluno) < cls.MIN_ALUNOS or n_itens < cls.MIN_ITENS:
            return None
        thetas = []
        for aid, arr in por_aluno.items():
            p = sum(arr) / n_itens
            p = min(max(p, 0.01), 0.99)
            thetas.append(math.log(p / (1 - p)))
        return float(np.mean(thetas))

class CorrectionService:
    @staticmethod
    def _gabarito_para_aluno(s, prova_id: int, lingua_aluno: str) -> dict:
        prova = s.get(Prova, prova_id)
        qs_lingua = parse_questoes_lingua(prova.questoes_lingua or "")
        linhas = s.query(Gabarito).filter_by(prova_id=prova_id).all()
        comum, ing, esp = {}, {}, {}
        for g in linhas:
            if g.lingua == LINGUA_COMUM:
                comum[g.numero_questao] = g.resposta
            elif g.lingua == "Inglês":
                ing[g.numero_questao] = g.resposta
            elif g.lingua == "Espanhol":
                esp[g.numero_questao] = g.resposta
        efetivo = {}
        for q in range(1, prova.numero_questoes + 1):
            if q in qs_lingua:
                if lingua_aluno == "Espanhol":
                    efetivo[q] = esp.get(q) or comum.get(q)
                else:
                    efetivo[q] = ing.get(q) or comum.get(q)
            else:
                efetivo[q] = comum.get(q)
        return efetivo

    @staticmethod
    def _corrigir_uma_imagem(prova_id, img, aluno_numero, conf_num, lingua_override, nome_arquivo, pagina=None):
        with Session() as s:
            prova = s.get(Prova, prova_id)
            if not prova:
                return {"erro": "Prova não encontrada"}
            aluno = s.query(Aluno).filter_by(numero=aluno_numero).first() if aluno_numero else None
            if prova.tem_lingua:
                lingua_aluno = lingua_override or (aluno.lingua_estrangeira if aluno else None) or "Inglês"
            else:
                lingua_aluno = LINGUA_COMUM
            gab = CorrectionService._gabarito_para_aluno(s, prova_id, lingua_aluno)
            marcacoes = OcrService.ler_marcacoes(img, prova.numero_questoes)
            qs_lingua = parse_questoes_lingua(prova.questoes_lingua or "")
            acertos, acertos_lingua, revisar = 0, 0, conf_num < 0.85
            linhas_resp = []
            for m in marcacoes:
                correta = (m["alternativa"] is not None and gab.get(m["questao"]) == m["alternativa"])
                if correta:
                    acertos += 1
                    if m["questao"] in qs_lingua:
                        acertos_lingua += 1
                if m["ambigua"] or m["confianca"] < 0.75:
                    revisar = True
                aplicada = lingua_aluno if m["questao"] in qs_lingua else LINGUA_COMUM
                linhas_resp.append(RespostaAluno(
                    prova_id=prova_id,
                    aluno_id=aluno.id if aluno else 0,
                    numero_questao=m["questao"],
                    resposta=m["alternativa"],
                    correta=correta,
                    confianca=m["confianca"],
                    lingua_aplicada=aplicada
                ))
            s.add_all(linhas_resp)
            s.add(Processamento(
                arquivo=nome_arquivo,
                status="OK" if not revisar else "Revisar",
                confianca=conf_num,
                pagina=pagina or 1,
                tipo="pdf" if pagina else "imagem"
            ))
            pct = 100.0 * acertos / max(prova.numero_questoes, 1)
            if aluno:
                res = s.query(Resultado).filter_by(aluno_id=aluno.id, prova_id=prova_id).first()
                if res:
                    res.acertos, res.percentual = acertos, pct
                else:
                    s.add(Resultado(aluno_id=aluno.id, prova_id=prova_id, acertos=acertos, percentual=pct))
            s.commit()
            return {
                "numero_aluno": aluno_numero,
                "confianca_numero": conf_num,
                "lingua_aplicada": lingua_aluno,
                "acertos": acertos,
                "acertos_lingua": acertos_lingua,
                "total": prova.numero_questoes,
                "precisa_revisao": revisar,
                "pagina": pagina
            }

    @staticmethod
    def corrigir_arquivo(prova_id: int, arquivo_bytes: bytes, nome_arquivo: str, lingua_override: Optional[str] = None):
        eh_pdf = nome_arquivo.lower().endswith(".pdf") or arquivo_bytes[:4] == b"%PDF"
        if eh_pdf:
            try:
                imgs = PdfService.pdf_para_imagens(arquivo_bytes, dpi=300)
            except Exception as e:
                return [{"erro": f"Falha ao ler PDF: {e}"}]
            resultados = []
            for i, img in enumerate(imgs, start=1):
                q = ImageProcessingService.qualidade(img)
                if q < 0.10:
                    resultados.append({"erro": f"Página {i}: qualidade insuficiente", "pagina": i})
                    continue
                numero, conf_num, _ = OcrService.ler_numero_aluno(img)
                r = CorrectionService._corrigir_uma_imagem(
                    prova_id, img, numero, conf_num, lingua_override,
                    nome_arquivo=f"{nome_arquivo}#pag_{i:03d}", pagina=i
                )
                resultados.append(r)
            return resultados
        else:
            arr = np.frombuffer(arquivo_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return [{"erro": "Imagem inválida"}]
            q = ImageProcessingService.qualidade(img)
            if q < 0.15:
                return [{"erro": "Imagem com qualidade insuficiente. Tire uma nova foto.", "qualidade": q}]
            numero, conf_num, _ = OcrService.ler_numero_aluno(img)
            return [CorrectionService._corrigir_uma_imagem(
                prova_id, img, numero, conf_num, lingua_override,
                nome_arquivo=nome_arquivo
            )]

    @staticmethod
    def corrigir(prova_id, imagem_bytes, lingua_override=None):
        return CorrectionService.corrigir_arquivo(prova_id, imagem_bytes, "captura.png", lingua_override)[0]

class ExportService:
    @staticmethod
    def excel(path="resultados_olimpo.xlsx"):
        wb = Workbook()
        ws = wb.active
        ws.title = "Resultados"
        cab = ["No", "Nome", "Turno", "Língua", "Humanas", "Natureza", "TRI"]
        ws.append(cab)
        for i, c_name in enumerate(cab, 1):
            c = ws.cell(1, i)
            c.fill = PatternFill("solid", fgColor=AZUL_MARINHO.replace("#", ""))
            c.font = Font(bold=True, color=DOURADO.replace("#", ""))
            c.alignment = Alignment(horizontal="center")
        with Session() as s:
            alunos = s.query(Aluno).order_by(Aluno.numero).all()
            provas = s.query(Prova).all()
            resultados = s.query(Resultado).all()
            for a in alunos:
                hum = next((r.acertos for r in resultados if r.aluno_id == a.id and any(p.id == r.prova_id and p.area == "Humanas" for p in provas)), 0)
                nat = next((r.acertos for r in resultados if r.aluno_id == a.id and any(p.id == r.prova_id and p.area == "Natureza" for p in provas)), 0)
                tri = next((r.tri for r in resultados if r.aluno_id == a.id and r.tri), "")
                ws.append([a.numero, a.nome, a.turno, a.lingua_estrangeira or "Inglês", hum, nat, tri or ""])

            wl = wb.create_sheet("Lingua Estrangeira")
            cab_l = ["No", "Nome", "Língua", "Acertos LE", "Total LE", "%"]
            wl.append(cab_l)
            for i, c_name in enumerate(cab_l, 1):
                c = wl.cell(1, i)
                c.fill = PatternFill("solid", fgColor=TURQUESA.replace("#", ""))
                c.font = Font(bold=True, color=AZUL_MARINHO.replace("#", ""))
            for a in alunos:
                total_le, acertos_le = 0, 0
                for p in provas:
                    if not p.tem_lingua:
                        continue
                    qs = parse_questoes_lingua(p.questoes_lingua or "")
                    total_le += len(qs)
                    for r in s.query(RespostaAluno).filter_by(aluno_id=a.id, prova_id=p.id).all():
                        if r.numero_questao in qs and r.correta:
                            acertos_le += 1
                pct = 100.0 * acertos_le / max(total_le, 1) if total_le else 0
                wl.append([a.numero, a.nome, a.lingua_estrangeira or "Inglês", acertos_le, total_le, round(pct, 1)])

            wd = wb.create_sheet("Detalhes")
            wd.append(["No", "Questão", "Resposta", "Correta", "Confiança", "Lingua"])
            for r in s.query(RespostaAluno).all():
                al = s.get(Aluno, r.aluno_id)
                wd.append([al.numero if al else 0, r.numero_questao, r.resposta or "", "OK" if r.correta else "X", round(r.confianca, 3), r.lingua_aplicada or LINGUA_COMUM])

            wp = wb.create_sheet("Processamento")
            wp.append(["Data", "Arquivo", "Página", "Tipo", "Status", "Confiança", "Erros"])
            for pr in s.query(Processamento).order_by(Processamento.id.desc()).all():
                wp.append([pr.data.strftime("%Y-%m-%d %H:%M"), pr.arquivo, pr.pagina, pr.tipo, pr.status, round(pr.confianca, 3), pr.erros or ""])

        wb.save(path)
        return path

    @staticmethod
    def csv(path="resultados_olimpo.csv"):
        with Session() as s:
            alunos = s.query(Aluno).order_by(Aluno.numero).all()
            df = pd.DataFrame([{"No": a.numero, "Nome": a.nome, "Turno": a.turno, "Língua": a.lingua_estrangeira or "Inglês"} for a in alunos])
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    @staticmethod
    def backup(path="olimpo_backup.zip"):
        with zipfile.ZipFile(path, "w") as z:
            if os.path.exists(DB_PATH):
                z.write(DB_PATH, "olimpo.db")
        return path

def gerar_prova_sintetica(numero_aluno, respostas, total_questoes=10, lingua_aluno="Inglês", questoes_lingua=(), alternativas=("A", "B", "C", "D", "E")):
    W, H = 800, 160 + total_questoes * 50
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    cv2.putText(img, "OLIMPO - CARTAO RESPOSTA", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    cv2.putText(img, "Aluno:", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, str(numero_aluno), (160, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, f"Lingua: {lingua_aluno}", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for q in range(1, total_questoes + 1):
        y = 140 + q * 50
        if q in questoes_lingua and q == min(questoes_lingua):
            cv2.putText(img, f"--- {lingua_aluno.upper()} ---", (30, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)
        cv2.putText(img, f"{q:02d}", (30, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        for j, alt in enumerate(alternativas):
            x = 120 + j * 80
            cv2.circle(img, (x, y), 15, (0, 0, 0), 2)
            cv2.putText(img, alt, (x - 8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if respostas.get(q) == alt:
                cv2.circle(img, (x, y), 13, (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()

def gerar_pdf_sintetico(lista_alunos: List[dict], total_questoes=10, questoes_lingua=(), gab_por_lingua: Dict[str, dict] = None):
    doc = fitz.open()
    for aluno in lista_alunos:
        img_bytes = gerar_prova_sintetica(
            aluno["numero"], aluno["respostas"],
            total_questoes=total_questoes,
            lingua_aluno=aluno.get("lingua", "Inglês"),
            questoes_lingua=questoes_lingua
        )
        img_path = f"_pg_{aluno['numero']}.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        rect = fitz.Rect(0, 0, 800, 160 + total_questoes * 50)
        page = doc.new_page(width=800, height=160 + total_questoes * 50)
        page.insert_image(rect, filename=img_path)
        if os.path.exists(img_path):
            os.remove(img_path)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def seed_demo():
    with Session() as s:
        if s.query(Aluno).count() == 0:
            for n in range(1, 61):
                turno = "Manhã" if n <= 20 else ("Tarde" if n <= 40 else "Noite")
                lingua = "Inglês" if n % 2 else "Espanhol"
                s.add(Aluno(numero=n, nome=f"Aluno {n:02d}", turno=turno, lingua_estrangeira=lingua))
            s.commit()
        if s.query(Prova).count() == 0:
            p = Prova(nome="Dia 1 - Simulado 01", area="Linguagens", numero_questoes=10, turma="3°A", tem_lingua=True, questoes_lingua="6-10")
            s.add(p)
            s.commit()
            for q, r in {1: "A", 2: "C", 3: "B", 4: "D", 5: "E"}.items():
                s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua=LINGUA_COMUM))
            for q, r in {6: "A", 7: "B", 8: "C", 9: "D", 10: "E"}.items():
                s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua="Inglês"))
            for q, r in {6: "B", 7: "C", 8: "D", 9: "E", 10: "A"}.items():
                s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua="Espanhol"))
            s.commit()
            log(f"Prova Dia 1 criada (id={p.id})")

seed_demo()

olimpo_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.yellow,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill=f"linear-gradient(135deg, {AZUL_MARINHO} 0%, {TURQUESA} 100%)",
    body_text_color=BRANCO,
    block_background_fill="#0E2A4A",
    block_border_color=DOURADO,
    button_primary_background_fill=DOURADO,
    button_primary_text_color=AZUL_MARINHO,
    button_secondary_background_fill="#1A3A63",
    button_secondary_text_color=TURQUESA_CLARO,
)

CSS = f"""
gradio-container {{ font-family: 'Segoe UI', sans-serif; }}
h1 {{ color: {DOURADO} !important; text-align:center; }}
.olimpo-header {{ text-align:center; padding: 10px; }}
"""

def ui_listar_alunos():
    with Session() as s:
        alunos = s.query(Aluno).order_by(Aluno.numero).all()
        return pd.DataFrame([{"No": a.numero, "Nome": a.nome, "Turno": a.turno, "Turma": a.turma, "Lingua": a.lingua_estrangeira or "Inglês"} for a in alunos])

def ui_cadastrar_aluno(numero, nome, turno, turma, lingua):
    try:
        with Session() as s:
            existente = s.query(Aluno).filter_by(numero=int(numero)).first()
            if existente:
                existente.nome = nome
                existente.turno = turno
                existente.turma = turma
                existente.lingua_estrangeira = lingua
                s.commit()
                return f"Aluno {numero} atualizado.", ui_listar_alunos()
            s.add(Aluno(numero=int(numero), nome=nome, turno=turno, turma=turma, lingua_estrangeira=lingua))
            s.commit()
            return f"Aluno {numero} cadastrado ({lingua}).", ui_listar_alunos()
    except Exception as e:
        return f"Erro: {e}", ui_listar_alunos()

def ui_definir_lingua_lote(ini, fim, lingua):
    ini, fim = int(ini), int(fim)
    with Session() as s:
        n = 0
        for num in range(ini, fim + 1):
            a = s.query(Aluno).filter_by(numero=num).first()
            if a:
                a.lingua_estrangeira = lingua
                n += 1
        s.commit()
        return f"{n} alunos ({ini} a {fim}) agora em {lingua}.", ui_listar_alunos()

def ui_criar_prova(nome, area, num_q, turma, tem_lingua, questoes_lingua):
    with Session() as s:
        p = Prova(nome=nome, area=area, numero_questoes=int(num_q), turma=turma, tem_lingua=bool(tem_lingua), questoes_lingua=questoes_lingua or "")
        s.add(p)
        s.commit()
        return f"Prova '{nome}' criada (id={p.id})."

def ui_ver_provas():
    with Session() as s:
        provas = s.query(Prova).order_by(Prova.id).all()
        return pd.DataFrame([{"ID": p.id, "Nome": p.nome, "Área": p.area, "Questões": p.numero_questoes, "Turma": p.turma, "LE?": "Sim" if p.tem_lingua else "Não", "Questões LE": p.questoes_lingua or "-", "Origem PDF": p.pdf_origem or "-"} for p in provas])

def ui_salvar_gabarito(prova_id, texto_comum, texto_ing, texto_esp):
    def parse(txt):
        itens = []
        for l in (txt or "").strip().splitlines():
            l = l.strip()
            if not l:
                continue
            partes = l.replace("-", " ").replace(":", "").split()
            if len(partes) >= 2 and partes[0].isdigit():
                itens.append((int(partes[0]), partes[1].upper()))
            elif len(partes) == 1 and len(partes[0]) == 1:
                itens.append((len(itens) + 1, partes[0].upper()))
        return itens

    with Session() as s:
        p = s.get(Prova, int(prova_id))
        if not p:
            return "Prova não encontrada."
        s.query(Gabarito).filter_by(prova_id=p.id).delete()
        n = 0
        for q, r in parse(texto_comum):
            s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua=LINGUA_COMUM, origem="manual"))
            n += 1
        for q, r in parse(texto_ing):
            s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua="Inglês", origem="manual"))
            n += 1
        for q, r in parse(texto_esp):
            s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua="Espanhol", origem="manual"))
            n += 1
        s.commit()
        return f"Gabarito salvo: {n} itens."

def ui_importar_gabarito_pdf(prova_id, pdf_file):
    if pdf_file is None:
        return "Envie um arquivo PDF.", None
    try:
        caminho = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
        with open(caminho, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        return f"Falha ao ler arquivo: {e}", None
    if not (pdf_bytes[:4] == b"%PDF"):
        return "O arquivo não é um PDF válido.", None

    por_lingua, texto, n_pag, usou_ocr = PdfService.parse_gabarito_pdf(pdf_bytes)
    if not any(por_lingua.values()):
        preview = texto[:500].replace("\n", " ")
        return f"Não consegui extrair respostas do PDF.\nPáginas: {n_pag} | OCR: {'sim' if usou_ocr else 'não'}\nTrecho: {preview}", None

    with Session() as s:
        p = s.get(Prova, int(prova_id))
        if not p:
            return "Prova não encontrada.", None
        for lingua, dic in por_lingua.items():
            if not dic:
                continue
            s.query(Gabarito).filter_by(prova_id=p.id, lingua=lingua).delete()
            for q, r in dic.items():
                s.add(Gabarito(prova_id=p.id, numero_questao=q, resposta=r, lingua=lingua, origem="pdf_ocr" if usou_ocr else "pdf_texto"))
        s.commit()

    linhas = [f"PDF lido: {n_pag} página(s) | OCR: {'sim' if usou_ocr else 'não'}"]
    for lingua, dic in por_lingua.items():
        if dic:
            amostra = ", ".join(f"{q}-{a}" for q, a in list(dic.items())[:10])
            extra = "..." if len(dic) > 10 else ""
            linhas.append(f"{lingua}: {len(dic)} questões ({amostra}{extra})")
    preview_txt = "\n".join(linhas)

    df = pd.DataFrame([
        {"Questão": q, "Língua": lingua, "Resposta": a}
        for lingua, dic in por_lingua.items()
        for q, a in dic.items()
    ]).sort_values(["Língua", "Questão"]) if por_lingua else pd.DataFrame()

    return preview_txt, df

def ui_corrigir(prova_id, arquivos, lingua_override):
    if not arquivos:
        return "Nenhum arquivo enviado.", ui_listar_alunos()
    resumo = []
    lingua_passada = None if lingua_override in (None, "Auto (pelo cadastro)") else lingua_override
    for f in arquivos:
        caminho = f.name if hasattr(f, "name") else f
        nome = os.path.basename(str(caminho))
        try:
            with open(caminho, "rb") as fp:
                data = fp.read()
        except Exception as e:
            resumo.append(f"{nome}: não foi possível ler ({e})")
            continue
        resultados = CorrectionService.corrigir_arquivo(int(prova_id), data, nome, lingua_override=lingua_passada)
        for r in resultados:
            if "erro" in r:
                pg = f" (pág. {r.get('pagina','?')})" if r.get("pagina") else ""
                resumo.append(f"{nome}{pg}: {r['erro']}")
                continue
            tag = "REVISAR" if r["precisa_revisao"] else "OK"
            pg = f" pág.{r['pagina']}" if r.get("pagina") else ""
            resumo.append(
                f"{tag} | {nome}{pg} | Aluno {r['numero_aluno']} "
                f"(conf {r['confianca_numero']:.0%}) | Língua: {r['lingua_aplicada']} "
                f"| Total: {r['acertos']}/{r['total']} | LE: {r['acertos_lingua']}"
            )
    return "\n".join(resumo), ui_listar_alunos()

def ui_testar_com_sinteticas(prova_id, qtd=6):
    with Session() as s:
        prova = s.get(Prova, int(prova_id))
        if not prova:
            return "Prova não encontrada."
        qs_lingua = parse_questoes_lingua(prova.questoes_lingua or "")

        def gabs(lingua):
            d = {}
            for g in s.query(Gabarito).filter_by(prova_id=prova.id).all():
                if g.numero_questao in qs_lingua:
                    if g.lingua == lingua:
                        d[g.numero_questao] = g.resposta
                else:
                    if g.lingua == LINGUA_COMUM:
                        d[g.numero_questao] = g.resposta
            return d

        msgs = []
        for n in range(1, qtd + 1):
            aluno = s.query(Aluno).filter_by(numero=n).first()
            lingua = aluno.lingua_estrangeira if aluno else "Inglês"
            gab = gabs(lingua)
            if not gab:
                continue
            resp = dict(gab)
            q_errada = list(gab.keys())[n % len(gab)]
            resp[q_errada] = "A" if gab[q_errada] != "A" else "B"
            img = gerar_prova_sintetica(n, resp, total_questoes=prova.numero_questoes, lingua_aluno=lingua, questoes_lingua=qs_lingua)
            r = CorrectionService.corrigir(prova.id, img)
            msgs.append(
                f"Aluno {n} ({lingua}) → {r.get('acertos','?')}/{r.get('total','?')} "
                f"| LE: {r.get('acertos_lingua','?')}/{len(qs_lingua)} "
                f"(conf {r.get('confianca_numero',0):.0%})"
            )
        return "\n".join(msgs)

def ui_testar_pdf_sintetico(prova_id, qtd=6):
    with Session() as s:
        prova = s.get(Prova, int(prova_id))
        if not prova:
            return None, "Prova não encontrada."
        qs_lingua = parse_questoes_lingua(prova.questoes_lingua or "")

        def gabs(lingua):
            d = {}
            for g in s.query(Gabarito).filter_by(prova_id=prova.id).all():
                if g.numero_questao in qs_lingua:
                    if g.lingua == lingua:
                        d[g.numero_questao] = g.resposta
                else:
                    if g.lingua == LINGUA_COMUM:
                        d[g.numero_questao] = g.resposta
            return d

        lista = []
        for n in range(1, qtd + 1):
            aluno = s.query(Aluno).filter_by(numero=n).first()
            lingua = aluno.lingua_estrangeira if aluno else "Inglês"
            gab = gabs(lingua)
            if not gab:
                continue
            resp = dict(gab)
            q_err = list(gab.keys())[n % len(gab)]
            resp[q_err] = "A" if gab[q_err] != "A" else "B"
            lista.append({"numero": n, "lingua": lingua, "respostas": resp})

        pdf_bytes = gerar_pdf_sintetico(lista, total_questoes=prova.numero_questoes, questoes_lingua=qs_lingua)
        pdf_path = "_demo_provas_sinteticas.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        resultados = CorrectionService.corrigir_arquivo(prova.id, pdf_bytes, "demo_provas.pdf")
        linhas = [f"PDF gerado: {len(lista)} páginas | {pdf_path}"]
        for r in resultados:
            if "erro" in r:
                linhas.append(f"pág.{r.get('pagina','?')}: {r['erro']}")
                continue
            tag = "OK" if not r["precisa_revisao"] else "REVISAR"
            linhas.append(
                f"{tag} pág.{r['pagina']} → Aluno {r['numero_aluno']} "
                f"({r['lingua_aplicada']}) | {r['acertos']}/{r['total']} "
                f"| LE: {r['acertos_lingua']} | conf {r['confianca_numero']:.0%}"
            )
        return pdf_path, "\n".join(linhas)

def ui_resultados():
    return ui_listar_alunos()

def ui_exportar_excel():
    return ExportService.excel()

def ui_exportar_csv():
    return ExportService.csv()

def ui_backup():
    return ExportService.backup()

with gr.Blocks(theme=olimpo_theme, css=CSS, title="Olimpo Correções v1.2") as demo:
    gr.HTML(
        f"""<div class="olimpo-header">
        <h1>OLIMPO CORREÇÕES v1.2</h1>
        <p style="color:{TURQUESA_CLARO}">Suporte completo a PDF (gabarito + provas) e Língua Estrangeira</p>
        </div>"""
    )

    with gr.Tab("Alunos"):
        with gr.Row():
            with gr.Column():
                n_num = gr.Number(label="Número (1-60)", value=1, precision=0)
                n_nome = gr.Textbox(label="Nome")
                n_turno = gr.Dropdown(["Manhã", "Tarde", "Noite", "Integral"], label="Turno", value="Manhã")
                n_turma = gr.Textbox(label="Turma")
                n_lingua = gr.Radio(LINGUAS, label="Língua Estrangeira", value="Inglês")
                btn_cad = gr.Button("Cadastrar / Atualizar aluno", variant="primary")
                msg_cad = gr.Markdown()
            with gr.Column():
                tabela_alunos = gr.Dataframe(value=ui_listar_alunos(), label="Alunos cadastrados")
        btn_cad.click(ui_cadastrar_aluno, [n_num, n_nome, n_turno, n_turma, n_lingua], [msg_cad, tabela_alunos])

        gr.Markdown("### Definir língua em lote")
        with gr.Row():
            lote_ini = gr.Number(label="De", value=1, precision=0)
            lote_fim = gr.Number(label="Até", value=30, precision=0)
            lote_lingua = gr.Radio(LINGUAS, label="Língua", value="Espanhol")
            btn_lote = gr.Button("Aplicar em lote", variant="secondary")
            msg_lote = gr.Markdown()
        btn_lote.click(ui_definir_lingua_lote, [lote_ini, lote_fim, lote_lingua], [msg_lote, tabela_alunos])

    with gr.Tab("Nova Prova"):
        p_nome = gr.Textbox(label="Nome da prova", value="Dia 1 - Simulado 01")
        p_area = gr.Dropdown(["Humanas", "Natureza", "Linguagens", "Matemática"], label="Área", value="Linguagens")
        p_qtd = gr.Number(label="Número de questões", value=10, precision=0)
        p_turm = gr.Textbox(label="Turma", value="3°A")
        p_tem_lingua = gr.Checkbox(label="Esta prova tem Língua Estrangeira?", value=True)
        p_questoes_lingua = gr.Textbox(label="Quais questões são de LE? (ex.: 6-10)", value="6-10")
        btn_prova = gr.Button("Criar prova", variant="primary")
        msg_prova = gr.Markdown()
        btn_prova.click(ui_criar_prova, [p_nome, p_area, p_qtd, p_turm, p_tem_lingua, p_questoes_lingua], [msg_prova])
        gr.Markdown("---")
        btn_listar = gr.Button("Listar provas", variant="secondary")
        tabela_provas = gr.Dataframe(value=ui_ver_provas(), label="Provas")
        btn_listar.click(ui_ver_provas, None, tabela_provas)

    with gr.Tab("Gabarito Manual"):
        g_id = gr.Number(label="ID da prova", value=1, precision=0)
        with gr.Row():
            g_comum = gr.Textbox(label="Comum", value="1 A\n2 C\n3 B\n4 D\n5 E", lines=8)
            g_ing = gr.Textbox(label="Inglês (só LE)", value="6 A\n7 B\n8 C\n9 D\n10 E", lines=8)
            g_esp = gr.Textbox(label="Espanhol (só LE)", value="6 B\n7 C\n8 D\n9 E\n10 A", lines=8)
        btn_gab = gr.Button("Salvar gabaritos", variant="primary")
        msg_gab = gr.Markdown()
        btn_gab.click(ui_salvar_gabarito, [g_id, g_comum, g_ing, g_esp], [msg_gab])

    with gr.Tab("Gabarito - via PDF"):
        gr.Markdown(
            """
            ### Importar gabarito diretamente de PDF
            - Aceita PDF **digitado** (extrai texto) ou **escaneado** (aplica OCR).
            - Reconhece automaticamente blocos **INGLÊS** / **ESPANHOL** e separa as respostas.
            - Formatos aceitos no texto: `1-A`, `1 A`, `01) A`, `Q1: A`, `ABCDE`, etc.
            - **Sobrescreve** apenas as línguas detectadas no PDF (Comum, Inglês, Espanhol).
            """
        )
        gp_id = gr.Number(label="ID da prova de destino", value=1, precision=0)
        gp_file = gr.File(label="Arquivo do gabarito (.pdf)", file_types=[".pdf"])
        btn_gp = gr.Button("Importar gabarito do PDF", variant="primary")
        msg_gp = gr.Textbox(label="Resumo da importação", lines=8)
        tabela_gp = gr.Dataframe(label="Respostas extraídas do PDF")
        btn_gp.click(ui_importar_gabarito_pdf, [gp_id, gp_file], [msg_gp, tabela_gp])

    with gr.Tab("Corrigir Provas"):
        c_id = gr.Number(label="ID da prova", value=1, precision=0)
        c_override = gr.Dropdown(
            ["Auto (pelo cadastro)", "Inglês", "Espanhol"],
            label="Língua Estrangeira (override)",
            value="Auto (pelo cadastro)",
            info="Use apenas quando a pilha inteira for de uma língua só."
        )
        c_files = gr.File(
            file_count="multiple",
            label="Imagens (.png/.jpg) E/OU PDFs (.pdf)",
            file_types=[".png", ".jpg", ".jpeg", ".pdf"]
        )
        btn_cor = gr.Button("Corrigir lote", variant="primary")
        msg_cor = gr.Textbox(label="Resultado do processamento", lines=18)
        btn_cor.click(ui_corrigir, [c_id, c_files, c_override], [msg_cor, tabela_alunos])

        gr.Markdown("---\n### Testes rápidos sem imagens reais")
        with gr.Row():
            c_demo_img = gr.Button("Gerar e corrigir 6 imagens sintéticas", variant="secondary")
            c_demo_pdf = gr.Button("Gerar PDF com 6 provas e corrigir", variant="secondary")
        msg_demo = gr.Textbox(label="Demo", lines=12)
        pdf_out = gr.File(label="PDF gerado (baixe para inspecionar)")

        c_demo_img.click(ui_testar_com_sinteticas, [c_id], [msg_demo])
        c_demo_pdf.click(ui_testar_pdf_sintetico, [c_id], [pdf_out, msg_demo])

    with gr.Tab("Resultados"):
        btn_res = gr.Button("Atualizar", variant="primary")
        tabela_res = gr.Dataframe(value=ui_resultados(), label="Resultados")
        btn_res.click(ui_resultados, None, tabela_res)

    with gr.Tab("Exportar / Backup"):
        with gr.Row():
            btn_x = gr.Button("Excel (.xlsx)", variant="primary")
            btn_c = gr.Button("CSV", variant="primary")
            btn_b = gr.Button("Backup (.zip)", variant="secondary")
        out_x = gr.File(label="Excel gerado")
        out_c = gr.File(label="CSV gerado")
        out_b = gr.File(label="Backup gerado")

        btn_x.click(ui_exportar_excel, None, out_x)
        btn_c.click(ui_exportar_csv, None, out_c)
        btn_b.click(ui_backup, None, out_b)

    with gr.Tab("Sobre"):
        gr.Markdown(
            f"""
            ### v1.2 - PDF + Língua Estrangeira
            - **Gabarito por PDF**: extrai texto ou aplica OCR e faz parsing robusto. Detecta blocos INGLÊS/ESPANHOL/Comum automaticamente.
            - **Provas por PDF**: cada página vira uma prova; conversão em 300 DPI.
            - **Detecção automática** imagem vs. PDF pelo conteúdo do arquivo.
            - **Auditoria**: cada PDF de origem fica registrado na tabela 'processamentos' com página, tipo e status.
            - **Excel** agora tem 4 abas: Resultados, Língua Estrangeira, Detalhes, Processamento.

            **Paleta**: `{AZUL_MARINHO}` → `{TURQUESA}` + `{DOURADO}`.
            """
        )

if __name__ == "__main__":
    demo.launch(share=True, debug=False)