"""Build a continuously typeset Chinese PDF (at most 15 pages)."""
from pathlib import Path
import os
import json
import html
import copy
from io import BytesIO
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image, Preformatted, Frame, BaseDocTemplate, PageTemplate
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
W,H=595.276,841.89
INK=colors.HexColor('#183746'); TEAL=colors.HexColor('#176B87')
LIGHT=colors.HexColor('#EDF4F7'); GRAY=colors.HexColor('#526570')

def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))

def register_fonts():
    regular=Path(os.environ.get('REPORT_FONT','C:/Windows/Fonts/simsun.ttc'))
    bold=Path(os.environ.get('REPORT_BOLD_FONT','C:/Windows/Fonts/simhei.ttf'))
    if not regular.exists():
        raise FileNotFoundError('Set REPORT_FONT to a Chinese TrueType font (.ttf/.ttc)')
    pdfmetrics.registerFont(TTFont('BodyCN',str(regular),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('BoldCN',str(bold if bold.exists() else regular),subfontIndex=0))
    pdfmetrics.registerFontFamily('BodyCN',normal='BodyCN',bold='BoldCN',italic='BodyCN',boldItalic='BoldCN')

class CodeBlock(Preformatted):
    """Shaded, line-splittable code instead of an indivisible one-cell table."""
    def wrap(self,availWidth,availHeight):
        width,height=super().wrap(availWidth,availHeight)
        self.height=height+12
        return width,self.height
    def draw(self):
        self.canv.saveState()
        self.canv.setFillColor(LIGHT)
        self.canv.rect(0,0,self.width,self.height,fill=1,stroke=0)
        self.canv.translate(9,-6)
        super().draw()
        self.canv.restoreState()
    def split(self,availWidth,availHeight):
        count=int((availHeight-12)//self.style.leading)
        if count<3:return []
        if len(self.lines)-count<3:count=len(self.lines)-3
        if count<3:return []
        return [CodeBlock('\n'.join(self.lines[:count]),self.style),
                CodeBlock('\n'.join(self.lines[count:]),self.style)]

class Report:
    def __init__(self,path,config):
        self.path=path
        self.config=config;self.n=0;self.contents=[];self.story=[]
        self.styles={
            'body':ParagraphStyle('body',fontName='BodyCN',fontSize=10.8,leading=15.7,textColor=INK,spaceAfter=5,wordWrap='CJK',allowWidows=0,allowOrphans=0),
            'small':ParagraphStyle('small',fontName='BodyCN',fontSize=9.5,leading=13,textColor=GRAY,spaceAfter=4,wordWrap='CJK',allowWidows=0,allowOrphans=0),
            'h2':ParagraphStyle('h2',fontName='BoldCN',fontSize=12.2,leading=18,textColor=TEAL,spaceBefore=6,spaceAfter=5,wordWrap='CJK',keepWithNext=True),
            'section':ParagraphStyle('section',fontName='BoldCN',fontSize=16,leading=23,textColor=INK,spaceBefore=12,spaceAfter=8,wordWrap='CJK',keepWithNext=True),
            'title':ParagraphStyle('title',fontName='BoldCN',fontSize=19,leading=27,textColor=INK,spaceBefore=0,spaceAfter=10,wordWrap='CJK',keepWithNext=True),
            'cell':ParagraphStyle('cell',fontName='BodyCN',fontSize=9.5,leading=13.6,textColor=INK,wordWrap='CJK'),
            'head':ParagraphStyle('head',fontName='BoldCN',fontSize=9.5,leading=13.6,textColor=colors.white,wordWrap='CJK'),
        }
    def p(self,text,style='body'):return Paragraph(text,self.styles[style])
    def h(self,text):return self.p(text,'h2')
    def table(self,rows,widths=None):
        data=[[self.p(html.escape(str(cell)), 'head' if i==0 else 'cell') for cell in row] for i,row in enumerate(rows)]
        t=Table(data,colWidths=widths or [503/len(rows[0])]*len(rows[0]),hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),TEAL),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
            ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#C3D7DF'))]))
        return t
    def fig(self,name,height=None):
        path=OUT/'figures'/name
        iw,ih=ImageReader(str(path)).getSize(); width=503;newh=width*ih/iw
        if height and newh>height:width*=height/newh;newh=height
        image=Image(str(path),width=width,height=newh);image.hAlign='CENTER';image.keepWithNext=True;return image
    def code(self,text):
        return CodeBlock(text,ParagraphStyle('code',fontName='Courier',fontSize=8.5,
            leading=10.8,textColor=INK,spaceBefore=4,spaceAfter=8))
    def page(self,title,items):
        # Introduce the findings before large figures, so a heading plus a
        # figure does not unnecessarily force 150-250 points of blank space.
        if title.startswith(('05 ', '10 ')):
            items=[items[2],items[0],items[1],*items[3:]]
        elif title.startswith('11 '):
            items=[items[5],items[2],items[3],items[4],items[0],items[1],*items[6:]]
        elif title.startswith('12 '):
            items=[items[2],items[3],items[4],items[5],items[0],items[1],*items[6:]]
        heading=self.p(title,'title' if not self.contents else 'section')
        heading.section_title=title
        heading.section_key=f'section{len(self.contents)}'
        self.story.extend([heading,*items]);self.contents.append(title)
    def finish(self):
        report=self
        class FlowDocument(BaseDocTemplate):
            def afterFlowable(self,flowable):
                if hasattr(flowable,'section_key'):
                    self.canv.bookmarkPage(flowable.section_key)
                    self.canv.addOutlineEntry(flowable.section_title,flowable.section_key,level=0)
        def draw_page(c,doc):
            c.saveState()
            c.setFillColor(TEAL);c.rect(0,H-8,W,8,fill=1,stroke=0)
            c.setFont('BodyCN',8.5);c.setFillColor(GRAY)
            c.drawString(46,H-29,'2026 秋  |  基于 Python 的问题解析  |  期末大作业')
            c.setStrokeColor(colors.HexColor('#CBDBE3'));c.line(46,43,W-46,43)
            c.setFont('BodyCN',8);c.drawString(46,28,f"{report.config['name']}  {report.config['student_id']}  ·  北京空气质量")
            c.drawRightString(W-46,28,f'{doc.page:02d} / {report.n:02d}' if report.n else f'{doc.page:02d}')
            c.restoreState()
        def build(destination):
            doc=FlowDocument(destination,pagesize=(W,H),title=self.config['title'],author=self.config['name'],
                subject='2026秋 基于Python的问题解析 期末大作业')
            frame=Frame(46,53,503,H-99,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
            doc.addPageTemplates(PageTemplate(id='continuous',frames=[frame],onPage=draw_page))
            doc.build(copy.deepcopy(self.story))
            return doc.page
        self.n=build(BytesIO())
        if self.n>15 and not os.environ.get('REPORT_LAYOUT_DRAFT'):
            raise ValueError(f'Report exceeds the 15-page limit: {self.n}')
        assert build(str(self.path))==self.n

def main():
    register_fonts()
    cfg=json.loads((ROOT/'project_config.json').read_text(encoding='utf-8'))
    dest=ROOT/'report';dest.mkdir(exist_ok=True)
    path=dest/f"{cfg['name']}{cfg['student_id']}期末大作业.pdf"
    a=read('data_audit.json');e=read('eda_summary.json');sel=read('model_selection.json');env=read('environment.json')
    scores=pd.read_csv(OUT/'metrics.csv').set_index('model');s=scores.loc[sel['selected']];base=scores.loc['Persistence']
    b=read('bootstrap.json');sens=read('sensitivity.json');sub=pd.read_csv(OUT/'subgroups.csv')
    pred=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['target_date'])
    missing_dates=pd.date_range('2016-03-01','2017-02-28').difference(pred.target_date.unique())
    coverage=100*len(pred)/(365*12)
    reduction=100*(1-s.RMSE/base.RMSE)
    r=Report(path,cfg);p,h,t,f=r.p,r.h,r.table,r.fig
    r.page('北京多站点空气质量分析与次日预测',[
        p('基于 12 站点小时数据的探索分析、机器学习比较与误差诊断','h2'),
        p(f"<b>作者：</b>{cfg['name']}　　<b>学号：</b>{cfg['student_id']}<br/><b>课程：</b>2026 秋《基于 Python 的问题解析》"),
        t([['数据规模','预测任务','验证方式'],['420,768 条小时记录\n12 站点 · 连续四年','日末预测下一日\n站点日均 PM2.5','三折时间交叉验证\n最后一年独立测试']]),
        Spacer(1,9),h('摘要'),
        p(f"本项目使用 UCI 北京多站点空气质量公开数据，围绕数据质量、时间变化、站点差异、气象关联与次日预测开展分析。以站点日为建模单位，规定至少18个有效小时，构造30项只使用当日及过去信息的特征。使用扩展时间窗口调参，比较均值基线、持续性基线、岭回归、随机森林和直方图梯度提升树。开发期交叉验证选择岭回归；在最后完整日历年的{len(pred):,}个可评估站点日上，其MAE为{s.MAE:.2f}、RMSE为{s.RMSE:.2f} μg/m³，R²为{s.R2:.3f}。相对于持续性基线，RMSE降低{reduction:.1f}%。"),
        p(f"探索性分析显示污染存在季节波动和跨站点同步性。7日块自助法给出相对持续性基线的RMSE改进区间为[{b[0]['ci95'][0]:.2f}, {b[0]['ci95'][1]:.2f}] μg/m³，多块长与覆盖门槛实验支持主要比较结果。移除晚间PM2.5使验证RMSE增加6.03 μg/m³。高浓度事件中，岭回归精确率为78.3%、召回率为45.0%；日变动小于50 μg/m³时，持续性基线误差更低。研究由此识别了日末信息的价值与模型在平稳、转折及高值场景中的具体权衡。"),
        h('研究问题'),
        p('<b>Q1 数据可信度：</b>缺失、异常和观测覆盖会怎样影响结果？<br/><b>Q2 污染规律：</b>时间、站点、气象与其他污染物呈现何种关联？<br/><b>Q3 预测增益：</b>机器学习能否超越“明天等于今天”？<br/><b>Q4 可靠性边界：</b>哪些场景易失准，特征增益是否稳定？'),
        p('关键词：PM2.5；时间验证；岭回归；事件诊断','small'),
        h('研究背景与具体任务'),
        p('日均PM2.5预测可以为下一日的监测值班、污染过程跟踪和风险提示提供量化参考。这个任务同时要求提取污染持续性、识别天气变化带来的转折，并理解极端高值的预测误差。本文以统一的数据清洗与时间验证流程，检验线性正则模型和树模型带来的预测增益。'),
        p('研究将总体浓度预测与高浓度事件识别连接起来：首先比较连续目标的MAE、RMSE和R²，再把固定阈值以上的预测转换为事件提示，计算精确率、召回率与误报数量。由此同时衡量典型日的预测质量和高值事件的识别代价。'),
    ])
    r.page('01  数据来源、变量与研究边界',[
        p('数据来自 UCI Beijing Multi-Site Air Quality [1]，记录2013-03-01至2017-02-28的12个北京监测站。污染观测与邻近气象站匹配，按小时组织。本文以历史站点观测为对象，分析浓度分布、跨站同步性与次日预测能力；区域指标采用有效站点等权平均。'),
        p('<b>官方页面：</b><link href="https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality" color="#176B87">archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality</link><br/><b>数据 DOI：</b>10.24432/C5RK5G　<b>许可：</b>CC BY 4.0。原始归档、来源说明和SHA256均保留在项目中。','small'),
        t([['字段组','原始字段','含义与单位'],
            ['索引/时间','No; year; month; day; hour; station','编号、日期、小时与站点。No不用作预测特征。'],
            ['污染物','PM2.5; PM10; SO2; NO2; CO; O3','各污染物浓度，单位均为 μg/m³。'],
            ['气象','TEMP; PRES; DEWP','气温(℃)、气压(hPa)、露点(℃)。'],
            ['降水与风','RAIN; wd; WSPM','降水(mm)、16方位风向、风速(m/s)。']], [72,185,246]),
        Spacer(1,9),h('读取与质量审计的实际结果'),
        t([['检查项','结果','处理'],['原始数据','420,768行 × 18列','仅合并内层归档的12站点CSV'],
            ['站点-小时重复 / 缺失时间键','0 / 0','保留完整网格；变量值仍可缺失'],
            ['负污染物、降水或风速','均为0条','规则仍写入校验代码'],
            ['按日聚合','17,532个站点日','12 × 1,461日'],
            ['PM2.5覆盖不足18小时','368个站点日','其日均PM2.5记为缺失'],
            ['可建模样本','16,962条','同时要求当天及次日PM2.5有效'],
            ['最终开发 / 间隔 / 测试','12,736 / 12 / 4,214条','按目标日期划分，保留时间间隔']], [146,137,220]),
        Spacer(1,8),p(f"测试日历跨度为365日，实际可评估日期为{pred.target_date.nunique()}日，站点日覆盖率为{coverage:.2f}%。没有可评估站点的日期为{', '.join(d.strftime('%Y-%m-%d') for d in missing_dates)}。评价总体由满足覆盖门槛的站点日构成。"),
        p('原始数据未经修改。处理后的daily.csv、model_table.csv包含保留的缺失标记、日期、目标和有效小时数，支持追溯每一步筛选。','small'),
    ])
    r.page('02  缺失值、异常值与清洗流程',[
        p(f"全样本PM2.5缺失率为{a['missing_pct']['PM2.5']:.2f}%，CO为{a['missing_pct']['CO']:.2f}%，气象数值变量均低于0.10%。时间戳完整不代表测量值完整。下面仅用开发期绘制缺失结构，避免据测试期分布修改方案。"),
        f('01_missing.png',255),p('图1　开发期的站点-变量缺失率。数字为百分比，颜色越深表示缺失越多。','small'),
        h('数据处理规则'),
        p('<b>① 先检查再聚合。</b>检查必要列、合法日期与重复站点小时键；重复键报错；负污染物、负降水、负风速转缺失并记录。负气温和负露点不属于错误。'),
        p('<b>② 按覆盖度聚合。</b>日均值至少18个有效小时；降水取有效小时之和；晚间PM2.5取18-23时均值且至少4小时；极值和标准差同样要求至少18小时。18小时对应75%的日覆盖度，是本项目设定的清洗参数。随后以20、24小时门槛重建数据并重新训练，检验其影响。'),
        p('<b>③ 保留极端污染观测。</b>开发期最高小时PM2.5为999 μg/m³。保留高值能够让模型和评价覆盖高污染事件；其仪器上限或测量异常属性需要设备元数据核实。箱线图隐藏离群点仅改变显示，训练使用完整的有效数值。'),
        p('<b>④ 训练内插补。</b>目标缺失不插补；当前PM2.5缺失也排除，以确保与持续性基线公平比较。其余特征的中位数及缺失指示只在每一训练折拟合，不用全样本均值或双向插值。'),
        p('<b>⑤ 保存审计结果。</b>data_audit.json保存每类计数、样本流转、数据哈希与验证边界；附加“至少20小时”的子集检查见“误差分层与覆盖度敏感性”一节。','small'),
    ])
    r.page('03  分布特征：均值背后的长尾',[
        f('02_distribution.png',220),p('图2　开发期有效站点日PM2.5分布。右图的对数仅用于观察，主模型在原始浓度尺度拟合。','small'),
        t([['描述统计（开发期）','值','统计口径'],
            ['小时均值',f"{e['dev_hourly_pm_mean']:.2f} μg/m³",'所有非缺失站点小时'],
            ['小时中位数',f"{e['dev_hourly_pm_median']:.2f} μg/m³",'对极端高值较不敏感'],
            ['小时95%分位数',f"{e['dev_hourly_pm_p95']:.2f} μg/m³",'用于描述小时分布的上尾'],
            ['有效站点日均值',f"{e['dev_daily_pm_mean']:.2f} μg/m³",'先按有效小时聚合，再按站点日等权'],
            ['次日目标90%分位数',f"{e['dev_target_p90']:.2f} μg/m³",'仅由开发期建模目标确定']], [157,125,221]),
        Spacer(1,9),h('从分布到建模选择'),
        p('均值高于中位数，说明少数高浓度时段明显拉高整体水平。小时统计按观测小时加权，站点日统计按合格站点日等权；日均化降低小时噪声，使研究聚焦于持续一天的污染水平。高浓度尾部对整体平方误差的贡献将在事件诊断中单独计算。'),
        p('RMSE对大误差更敏感，适合让高浓度日的严重预测偏差进入模型选择；MAE便于解释典型绝对误差。二者同时报告。没有使用MAPE：低浓度时分母很小，可能放大并扭曲相对误差。'),
        p('主模型在原始浓度尺度拟合，预测值统一截断至非负，交叉验证和测试使用相同规则。高浓度分组阈值取开发目标90%分位数，定义本项目的经验高值事件；与按法规限值或AQI分级的任务需要分别设定评价标准。'),
    ])
    r.page('04  时间变化：季节性与共同波动',[
        f('03_time.png',337),p('图3　上：至少8站有效时的等权日均值与30日后向滚动均值；下：不同月份的站点日均值和中位数。仅使用开发期。','small'),
        h('观察与解释'),
        p(f"开发期的12月平均日浓度为{e['month_daily_mean']['12']:.2f} μg/m³，8月为{e['month_daily_mean']['8']:.2f} μg/m³。月度统计呈现秋冬及部分春季偏高的总体格局，日尺度仍保留显著起伏。季节特征负责描述背景水平，近期状态特征负责追踪事件变化。"),
        p('时间图中的短期起伏远大于平滑曲线所展示的缓慢变化。模型因此同时使用季节性日历特征和近期污染状态：年内日期的正弦、余弦用于平滑表达年周期；当日均值、晚间均值、过去1/2/7日及3/7日滚动平均刻画污染持续性与变化。'),
        p(f"当前日均PM2.5与次日目标的Spearman相关为{e['pm_persistence_spearman']:.3f}。这说明持续性基线有依据，但相关性并不接近1，天气转换、输送与未观测因素仍可能使浓度快速升降。"),
        p('月度汇总混合多个年份，时间曲线同时受到天气与排放等因素影响。本文使用这些统计描述预测背景；政策效果需要额外的干预信息和对照设计。','small'),
    ])
    r.page('05  站点差异与气象关联',[
        f('04_stations.png',233),p('图4　开发期站点日均浓度与站点间Spearman相关。站点差异同时受地点、缺测和时间构成影响。','small'),
        p(f"有效站点日均值最高为东四（{e['station_daily_mean']['Dongsi']:.2f}），最低为定陵（{e['station_daily_mean']['Dingling']:.2f} μg/m³）。多数站点具有共同波动，因此同一日期的站点观测并非独立样本；验证与自助抽样均需保留日期结构。"),
        f('05_weather.png',222),p('图5　日均变量相关矩阵与不同风速组的PM2.5分布。相关系数按变量对的有效观测计算；风速箱线图不显示离群点。','small'),
        p(f"PM2.5与PM10的Spearman相关为{e['pm_pm10_spearman']:.3f}，与风速为{e['pm_wind_spearman']:.3f}。较高风速组的PM2.5分布整体较低，这一观测关联与扩散解释相容。季节、排放与天气共同变化，相关矩阵用于描述特征联系和构建预测输入。"),
        p('站点比较覆盖这12处监测点；气象观测按邻近气象站匹配。站点共享气象与区域污染背景，进一步支持按日期整体划分和抽样的设计。','small'),
    ])
    r.page('06  任务定义与特征工程',[
        p('<b>预测时点：</b>第t日全部观测完成之后。<b>目标：</b>第t+1日同一站点的平均PM2.5。<b>信息集：</b>只包含第t日及更早观测，以及已知的次日日历。假定观测及时获得；实际发布延迟需在部署时另行处理。'),
        r.code('X[s, t] = features(observations at dates <= t)\ny[s, t] = mean_PM25[s, t + 1 day]\nprediction[s, t + 1] = max(0, fitted_model(X[s, t]))'),
        t([['特征组','数量','内容'],
            ['站点与日历','5','站点、次日星期、年内日期sin/cos、线性时间趋势'],
            ['PM2.5状态与历史','11','当日均值/有效小时/最大值/标准差/晚间均值；滞后1、2、7日；3、7日后向均值；与前一日的差'],
            ['气象','8','当日气温、气压、露点、降水、风速、风向sin/cos及气压一日差'],
            ['其他污染与区域信息','6','当日PM10、SO2、NO2、CO、O3；至少8站有效时的当日PM2.5区域等权均值']], [104,40,359]),
        Spacer(1,8),p('合计30项原始特征。数值特征做训练内中位数插补、缺失指示和标准化；站点/星期做独热编码。编码后的列数会增加。树模型不依赖标准化，但统一处理使实验流程一致；所有转换器都只在训练数据拟合。[2]'),
        h('时间对齐与风向编码'),
        p('<b>严格次日：</b>先构造完整站点日网格，再按站点shift(-1)生成目标；否则缺失日期可能让“下一行”跳过一天。<br/><b>后向窗口：</b>滚动均值包含t日及过去，不用中心窗口；t日均值预测t+1合法，t+1日天气和污染物不合法。<br/><b>方向变量：</b>将16方位转换成角度的sin/cos后求日均，避免北附近方位的数值跳变。'),
        p('测试年逐日获得新观测，再预测下一日；模型参数保持冻结。该一步滚动预测覆盖完整测试日历年，其中363日有合格评价样本。每次预测都会更新输入历史，保持与日末发布场景一致。'),
        p('区域均值使用同一t日各站信息，适用于观测共享的多站点系统；若某站需要完全独立部署，应移除该特征并重新验证。','small'),
    ])
    r.page('07  时间验证、数据泄漏与评价指标',[
        p('开发目标日期为2013-03-02至2016-02-28；2016-02-29留作目标日期间隔；2016-03-01至2017-02-28为最终测试区间。最后一年包含四季，避免只用冬季几个月评价模型。模型与参数均在开发期决定。'),
        t([['折','训练目标截止','验证目标区间','训练 / 验证数'],
            ['1','2015-02-27','2015-03-01 至 06-30','8,442 / 1,421'],
            ['2','2015-06-29','2015-07-01 至 10-31','9,863 / 1,438'],
            ['3','2015-10-30','2015-11-01 至 2016-02-28','11,301 / 1,423']], [30,114,230,129]),
        Spacer(1,10),p('每折训练从2013-03-02开始，随时间扩展；训练标签最大日期严格早于验证特征最小日期。同一日期的12站全部划入同一侧，不使用普通随机拆分，以免相似污染事件同时进入训练和验证。'),
        r.code('train = target_date < valid_start - pd.Timedelta(days=1)\nvalid = (target_date >= valid_start) & (target_date < valid_end)\nassert train_target_dates.max() < valid_feature_dates.min()\n# Fit preprocessing only inside each training fold.'),
        h('用开发期选择模型，用独立年份评价'),
        p('用GridSearchCV进行三折搜索，以折RMSE的算术平均为主选择标准，同时保存MAE和训练分数。最优模型在全部开发数据上重新拟合，再对测试年评价。补充的稳健性与事件分析沿用已经选定的模型和参数。'),
        t([['指标','定义','用途与注意点'],
            ['MAE','mean(|预测 - 实际|)','典型绝对误差；单位μg/m³。'],
            ['RMSE','sqrt(mean((预测 - 实际)²))','主指标；更重视大误差。'],
            ['R²','1 - SSE/SST','相对测试目标均值的解释度；可能为负。'],
            ['Bias','mean(预测 - 实际)','正值表示高估，负值表示低估。']], [65,205,233]),
        Spacer(1,7),p('三折标准差反映季节与时期差异，不是独立同分布误差的置信区间；最终不确定性使用保留日期相关的块自助法。测试集的平均水平、分位数或标签不参与插补、编码、阈值确定与调参。','small'),
    ])
    tune=sel['tuning']
    r.page('08  模型原理与实际调参过程',[
        p('<b>均值基线：</b>始终预测训练目标均值。<b>持续性：</b>用当日PM2.5预测次日。<b>岭回归：</b>最小化残差平方和加α乘系数平方和，缓解共线性和参数不稳定。<b>随机森林：</b>多棵随机化树平均，学习非线性和交互。<b>HGB：</b>直方图梯度提升树逐步修正残差，以学习率和树复杂度控制拟合。[3-5]'),
        t([['方法','候选参数','最优 / 组合数'],
            ['Ridge','α = 0.1, 1, 10, 100','α=10；4组'],
            ['随机森林','树数160；深度10/不限；叶最少样本3/10；特征比例0.8','深度10；叶样本10；4组'],
            ['HGB','轮数240；学习率0.05/0.1；叶数15/31；L2=0/10；关闭内部早停','学习率0.05；叶数31；L2=10；8组']], [77,292,134]),
        Spacer(1,7),p('共16组候选 × 3折 = 48次候选拟合，另含最优模型重拟合等步骤。随机种子42。HGB关闭随机划分的内部早停，避免与时间验证设计冲突。候选的全部参数、逐折分数与训练时间保存在cv_*.csv。','small'),
        f('06_tuning.png',189),p('图6　每个算法按参数网格顺序排列的候选。实线为验证RMSE及折间标准差，虚线为训练RMSE；候选序号对应结果CSV的行顺序。','small'),
        t([['方法','首组CV RMSE','最优CV RMSE','三折RMSE'],
            *[[name.replace('RandomForest','RF').replace('HistGradientBoosting','HGB'),
               f"{tune[name]['default_grid_first_rmse']:.2f}",f"{tune[name]['cv_rmse']:.2f}",
               ' / '.join(f'{v:.2f}' for v in tune[name]['fold_rmse'])] for name in tune]], [77,112,112,202]),
        Spacer(1,7),p('首组是网格中的参照候选，不称为库的默认参数。岭回归调参增益很小，说明其结果对本次α范围较稳定；三种模型的第三折都明显更难，提示季节和时期差异比小幅调参影响更大。'),
    ])
    r.page('09  独立测试结果与不确定性',[
        p(f"交叉验证选择<b>{sel['selected']}</b>，测试年继续取得最低RMSE。所有模型在相同{len(pred):,}个站点日上评分。其R²为{s.R2:.3f}，衡量相对于测试目标均值常数的平方误差改善；MAE与RMSE分别描述典型绝对误差和大误差。"),
        t([['方法','CV RMSE','测试MAE','测试RMSE','测试R2','Bias'],
            *[[name.replace('HistGradientBoosting','HGB').replace('RandomForest','RF').replace('TrainingMean','均值').replace('Persistence','持续性'),
               f'{row.CV_RMSE:.2f}',f'{row.MAE:.2f}',f'{row.RMSE:.2f}',f'{row.R2:.3f}',f'{row.Bias:.2f}'] for name,row in scores.iterrows()]], [88,83,83,83,83,83]),
        Spacer(1,5),f('07_models.png',202),p('图7　测试年整体误差；橙色为按开发期验证选定的模型。误差单位为μg/m³。','small'),
        h('配对时间块自助法'),
        p('将每天各站的平方误差先汇总，再按7个连续日的块抽样；每次保留同一天全部站点，共1,500次。差值定义为“参照模型RMSE - 岭回归RMSE”，正值有利于岭回归。没有可评估样本的日期保留为空权重日。'),
        t([['比较','RMSE改进','95%百分位区间'],
            *[[x['reference'],f"{x['rmse_improvement']:.2f}",f"[{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"] for x in b]], [190,125,188]),
        Spacer(1,7),p(f"岭回归相对持续性的RMSE下降{reduction:.1f}%，且此区间不跨0，说明改进在该测试期具有一定稳定性。区间只描述给定模型和该历史测试期的评估不确定性；7日块长是一项假设，不覆盖重训练随机性、长周期相关或未来年份漂移。"),
    ])
    r.page('10  预测曲线与残差诊断',[
        f('08_residuals.png',212),p('图8　左：测试样本预测-实际值的六边形密度图；右：残差随实际浓度变化。虚线分别表示理想预测与零误差。','small'),
        p('高浓度处的点普遍低于理想对角线，残差向负值扩展，表明整体无明显偏差并不意味着各浓度区间都无偏。平方损失学习的是条件平均；当缺少明日天气与排放变化信息时，突发高峰容易被低估。'),
        f('09_predictions.png',284),p('图9　上为整个测试年的各站点等权平均；下为2016年12月至2017年1月冬季窗口。两条曲线在相同可评估站点上平均。','small'),
        p('模型能跟随部分升降趋势，但存在峰值压缩和转折滞后。区域平均会抵消部分站点误差，因此图9仅用于展示趋势；总体结果表的主评价仍在原始站点日层面计算，不能用平均曲线的视觉吻合替代样本级误差。'),
    ])
    high=sub[(sub.dimension=='level') & sub.group.str.startswith('High')].iloc[0]
    r.page('11  误差分层与覆盖度敏感性',[
        f('10_subgroups.png',205),p(f"图10　按目标季节和开发目标90%分位阈值（{e['dev_target_p90']:.2f} μg/m³）分组。该阈值只用于经验分层。",'small'),
        t([['组别','样本数','岭回归RMSE','持续性RMSE','岭回归Bias'],
            *[[row.group,str(row.n),f'{row.RMSE:.2f}',f'{row.persistence_RMSE:.2f}',f'{row.Bias:.2f}'] for row in sub[sub.dimension.isin(['season','level'])].itertuples()]], [129,61,104,104,105]),
        Spacer(1,8),p(f"冬季绝对误差明显大于夏季。高浓度组共有{high.n}个站点日，其RMSE为{high.RMSE:.2f}、Bias为{high.Bias:.2f} μg/m³，显示较强低估。该组R²为{high.R2:.3f}，意味着在这一条件子集上表现弱于用该子集真实均值作事后常数；这种常数在实际预测时不可直接获得。"),
        p('站点RMSE从定陵的41.27到古城的50.39 μg/m³不等；12站上岭回归均优于持续性基线。各站目标分布、缺测和事件强度共同构成误差差异，完整分组结果保存在subgroups.csv。'),
        h('把有效小时要求提高到20：同一模型的子集复评'),
        p(f"要求当日和目标日均至少20小时后，保留{sens['n']:,}个测试样本；原岭回归模型RMSE为{sens['selected']['RMSE']:.2f}，持续性为{sens['persistence']['RMSE']:.2f} μg/m³。该子集复评保持了总体排序。第14节进一步从小时数据重新聚合、构造特征并训练，区分样本筛选与模型训练的影响。"),
    ])
    abl=pd.read_csv(OUT/'ablation.csv')
    r.page('12  特征贡献与模型优劣分析',[
        f('11_interpretation.png',245),p('图11　左为最后一折验证集上重复5次的单变量置换重要性；右为固定岭回归算法和α=10的三折特征消融。误差单位μg/m³。','small'),
        p('晚间PM2.5置换后的误差增量最大，说明临近日末的状态对次日均值预测具有较强信息。PM2.5当日极值、均值和风速等也有贡献。置换会破坏变量间的真实联系，相关特征之间还会分摊贡献，因此这些数值不等于独立因果效应，也不宜作为直接干预排序。'),
        t([['特征方案','数量','平均CV RMSE','解释'],
            ['历史 + 日历','16',f'{abl.iloc[0].cv_rmse:.2f}','污染持续性已包含主要信号'],
            ['再加入当日气象','24',f'{abl.iloc[1].cv_rmse:.2f}','误差有所降低'],
            ['全部特征','30',f'{abl.iloc[2].cv_rmse:.2f}','增加变量未继续改善平均CV']], [143,40,112,208]),
        Spacer(1,7),p('“气象方案”平均CV RMSE比全部特征低约0.08 μg/m³。固定参数的消融表明，额外污染物在三折平均中没有进一步降低误差；不同时间折的表现提示其增益具有时期差异。该比较为简化输入提供线索，主评价沿用初次验证选定的全特征模型。'),
        h('模型表现差异'),
        p('<b>岭回归：</b>训练快、约束强，适合共线的污染历史特征；但线性关系难以表达复杂交互，对突发峰值仍不足。<br/><b>随机森林：</b>可表达非线性且较稳健，但叶节点均值限制了对稀少高值的外推；树间平均会平滑峰值。<br/><b>HGB：</b>能学习更复杂模式，却更依赖本次特征、参数范围和训练期分布；测试Bias较负，存在整体低估。'),
        p('模型排序对应本次特征、时间切分和16组候选搜索预算。下一节进一步移除晚间与区域信息，直接检验模型对两项关键输入的依赖。','small'),
    ])
    add_extended_sections(r)
    r.page('17  Python实现：类、异常与函数调用',[
        p('项目分为数据、建模、绘图三个模块，由run_analysis.py组织。数据集类维护路径、覆盖规则与审计状态；自定义异常说明输入失败原因；独立函数完成特征工程、评价和验证。主要接口如下。'),
        r.code('class DataValidationError(ValueError):\n    """Input violates the data contract."""\n\nclass AirQualityDataset:\n    def __init__(self, archive, min_hours=18):\n        self.archive = Path(archive)\n        if not isinstance(min_hours, int) or not 1 <= min_hours <= 24:\n            raise ValueError("min_hours must be an integer in [1, 24]")\n        self.min_hours = min_hours\n        self.audit = {}\n\n    def load(self):\n        if not self.archive.is_file():\n            raise FileNotFoundError(f"Data archive not found: {self.archive}")\n        # Read nested ZIP, validate schema and station-hour keys.\n        # BadZipFile / ParserError are wrapped as DataValidationError.'),
        p('<b>对象与函数的调用链：</b>数据集类维护聚合规则；特征、时间划分、调参和指标分别由独立函数实现。','small'),
        r.code('data = AirQualityDataset(ROOT / "data/raw/beijing-air-quality.zip")\nhourly = data.load()\ndaily = data.aggregate(hourly)\nframe, feature_groups = build_features(daily)\nfolds, descriptions = calendar_folds(dev)\nmodels, tuning = tune_models(dev, features, folds, out)\nscore = metrics(test.target, model.predict(test[features]))\n\nif __name__ == "__main__":\n    try:\n        main()\n    except (FileNotFoundError, ValueError) as exc:\n        raise SystemExit(f"Analysis stopped: {exc}") from exc'),
        p('数据归档缺失、ZIP损坏或必要字段不合法时，程序通过明确异常停止，并保留原始错误链。PersistenceRegressor和NonnegativeRegressor展示了继承、统一fit/predict接口与模型组合。'),
        h('自动验证的关键内容'),
        p('① 目标等于严格下一日的真实日均值；② 修改未来数据不会改变过去特征；③ 同一天站点不跨训练/验证；④ 训练标签早于验证特征日期；⑤ 特征不含目标列；⑥ 损坏或缺失归档给出明确异常；⑦ 从保存的逐行预测重算所有指标；⑧ 评价覆盖与筛选规则一致。'),
    ])
    repo=cfg.get('github_url','').strip()
    repo_text=f'<link href="{html.escape(repo)}" color="#176B87">{html.escape(repo)}</link>' if repo else '未配置项目仓库地址。'
    r.page('18  结论、复现与参考资料',[
        p(f"<b>结论一：</b>按日覆盖门槛聚合、训练内预处理和严格次日对齐，形成可追溯的数据分析流程。<br/><b>结论二：</b>近期污染状态、季节波动与跨站同步性构成主要预测信息；移除晚间PM2.5使平均验证RMSE上升约6.03 μg/m³。<br/><b>结论三：</b>岭回归在测试年RMSE为{s.RMSE:.2f} μg/m³，相对持续性降低{reduction:.1f}%；多块长和覆盖门槛实验支持该结果的稳定性。<br/><b>结论四：</b>岭回归减少高浓度误报，提高精确率，同时降低召回率；日变动较小时持续性表现更好，快速升降时岭回归降低误差。"),
        p('<b>适用范围与改进方向：</b>结果覆盖2013-2017年12站历史观测，解释单位为合格站点日。模型假定当天观测及时到齐，缺失机制和未来年份分布变化仍会影响部署。后续可引入有真实发布时间的天气预报、滚动重训练和分位数预测；高事件召回率应作为独立验证目标，与误报成本共同优化。气象关联采用观测性解释，政策效应需要额外对照设计。'),
        h('项目与复现'),p(f'<b>GitHub：</b>{repo_text}','small'),
        p(f"<b>运行环境：</b>Python {env['python']}；NumPy {env['packages']['numpy']}；pandas {env['packages']['pandas']}；Matplotlib {env['packages']['matplotlib']}；scikit-learn {env['packages']['scikit-learn']}。依赖固定于requirements.txt；种子42。原始归档、处理后数据、全部调参/预测结果和图表均随项目提供。",'small'),
        r.code('python -m pip install -r requirements.txt\npython download_data.py\npython run_analysis.py\npython extended_analysis.py\npython -m unittest discover -s tests -v\npython build_report.py'),
        p('项目入口为README.md；Notebook为analysis.ipynb；主要结果为results/metrics.csv。','small'),
        h('参考资料（数据与官方方法文档）'),
        p('[1] Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository. <link href="https://doi.org/10.24432/C5RK5G" color="#176B87">doi.org/10.24432/C5RK5G</link>。许可：CC BY 4.0。','small'),
        p('[2] scikit-learn 1.7.2. Pipeline. <link href="https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html" color="#176B87">官方Pipeline文档</link>。\n[3] scikit-learn 1.7.2. Ridge. <link href="https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.Ridge.html" color="#176B87">官方岭回归文档</link>。','small'),
        p('[4] scikit-learn 1.7.2. RandomForestRegressor. <link href="https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestRegressor.html" color="#176B87">官方随机森林文档</link>。\n[5] scikit-learn 1.7.2. HistGradientBoostingRegressor. <link href="https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html" color="#176B87">官方梯度提升树文档</link>。','small'),
    ])
    r.finish()
    reader=PdfReader(path)
    if len(reader.pages)>15:raise ValueError('Page count check failed')
    (OUT/'report_verification.json').write_text(json.dumps({'pages':len(reader.pages),
        'section_titles':r.contents,'layout':'continuous','github_url':repo,'github_published':bool(repo),
        'file':str(path.name)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)

def add_extended_sections(r):
    p,h,t,f=r.p,r.h,r.table,r.fig
    avail=pd.read_csv(OUT/'feature_availability.csv')
    cov=pd.read_csv(OUT/'coverage_retraining.csv')
    common=pd.read_csv(OUT/'coverage_common_sample.csv')
    events=pd.read_csv(OUT/'high_event_diagnostics.csv').set_index('model')
    changes=pd.read_csv(OUT/'change_errors.csv')
    monthly=pd.read_csv(OUT/'monthly_errors.csv')
    ext=read('extended_summary.json');blocks=read('block_sensitivity.json')
    r.page('13  晚间与区域信息的作用',[
        p('置换重要性反映拟合后模型对变量的依赖；删除变量后重新拟合，则允许其他输入补偿丢失的信息。本实验在开发期三折内固定岭回归α=10，分别移除晚间PM2.5、区域均值或两者，直接评价数据到齐条件改变后的误差。'),
        f('15_availability.png',157),p('图12　固定算法、参数与验证边界的输入可获得性实验。每个方案均在各训练折重新拟合预处理和回归模型。','small'),
        p(f"完整输入的平均CV RMSE为{avail.iloc[0].CV_RMSE:.2f}；移除晚间均值后上升至{avail.iloc[1].CV_RMSE:.2f}，增加{avail.iloc[1].CV_RMSE-avail.iloc[0].CV_RMSE:.2f} μg/m³。第三折由68.71升至79.84，说明日末状态在冬季验证期具有更大预测价值。晚间浓度能够提供比全天均值更接近发布时点的状态信息，这与污染过程的日内变化相符。"),
        p(f"单独移除区域均值后，平均误差为{avail.iloc[2].CV_RMSE:.2f}，与完整模型接近；两项同时移除时为{avail.iloc[3].CV_RMSE:.2f}。因此，在此验证设计下，优先确保本站晚间观测到齐，比维护区域均值输入更有价值。若需要提前到白天发布预测，应重新定义日内统计窗口，再开展时间验证。"),
    ])
    r.page('14  稳健性：清洗门槛与时间块长度',[
        p('稳健性分析分别改变数据覆盖门槛和评估抽样块长，检查主要模型排序对两项分析选择的依赖。补充实验在初次测试完成后开展，沿用已经选定的岭回归和α=10，主模型保持固定。'),
        t([['小时门槛','训练数 / 测试数','各自样本RMSE','共同样本RMSE'],
            *[[int(row.min_hours),f'{int(row.n_train):,} / {int(row.n_test):,}',
               f'{row.RMSE:.2f}',f"{common.loc[common.min_hours==row.min_hours,'RMSE'].iloc[0]:.2f}"] for row in cov.itertuples()]], [73,165,132,133]),
        Spacer(1,7),p(f"每个门槛都从小时数据重新聚合、生成滞后和滚动特征，再用该门槛对应的开发数据重新训练。为区分样本变化和训练变化，另取三方案共同的{int(common.n_common.iloc[0]):,}个测试站点日评分。24小时方案的测试覆盖率由96.21%降至72.76%；共同样本RMSE为47.98，18小时方案为47.71。更严格门槛减少可用信息，且未改善此处的共同样本误差。"),
        f('12_robustness.png',207),p('图13　左：不同时间块长下的配对RMSE改善与95%百分位区间；右：覆盖门槛改变后的完整重训与共同样本比较。','small'),
        p('以1、3、7、14、28日为块，各进行1,500次抽样，同一天的站点共同抽取。相对持续性与随机森林的改进区间在五种块长下均保持正值。28日块时，改进区间分别为[12.85, 27.49]和[2.03, 9.55] μg/m³，支持所选模型在该测试年的比较结果。'),
        p('抽样区间衡量给定拟合模型在本测试期的误差差异；重训练覆盖实验则反映清洗与训练样本变化的影响。两类证据分别对应评价不确定性和数据处理敏感性。'),
    ])
    ridge=events.loc['Ridge'];baseline=events.loc['Persistence']
    r.page('15  高浓度事件：精确率与召回率权衡',[
        p(f"将实际与预测日均PM2.5分别与开发期90%分位阈值{ext['threshold']:.2f} μg/m³比较，形成高浓度事件标签。测试集中有{ext['high_n']}个高值站点日，占{100*ext['high_fraction']:.2f}%，却贡献岭回归总平方误差的{100*ext['high_sse_fraction']:.2f}%。这说明高值尾部是改进整体RMSE的重点。"),
        f('13_events.png',201),p('图14　固定浓度阈值下的事件识别指标与岭回归混淆矩阵。标签按站点日定义；同日多个站点可同时出现事件。','small'),
        t([['模型','精确率','召回率','F1','误报 / 漏报'],
            *[[name.replace('HistGradientBoosting','HGB').replace('RandomForest','RF').replace('Persistence','持续性'),
               f'{row.precision:.3f}',f'{row.recall:.3f}',f'{row.f1:.3f}',f'{int(row.fp)} / {int(row.fn)}'] for name,row in events.iterrows()]], [100,88,88,88,139]),
        Spacer(1,7),p(f"岭回归发出{int(ridge.n_alerts)}次高浓度提示，命中{int(ridge.tp)}次，误报{int(ridge.fp)}次，精确率为{ridge.precision:.3f}；实际433次事件中识别195次，召回率为{ridge.recall:.3f}。持续性提示435次，误报206次，召回率为{baseline.recall:.3f}。岭回归降低误报并提高F1，同时增加34次漏报，体现整体回归损失与事件召回目标的权衡。"),
        p('精确率=TP/(TP+FP)，召回率=TP/(TP+FN)，F1为二者的调和平均。这里直接对既有回归输出应用同一经验阈值。若实际业务更重视漏报，可在新的验证窗口联合比较阈值、非对称损失和专门分类器，并报告每增加一次命中所需承担的误报。'),
    ])
    worst=monthly.loc[monthly.RMSE.idxmax()];best=monthly.loc[monthly.RMSE.idxmin()]
    r.page('16  浓度变化与预测误差',[
        p(f"逐月评分进一步展示测试期的误差结构。岭回归误差最高月份为{worst.month}，RMSE为{worst.RMSE:.2f}；最低月份为{best.month}，RMSE为{best.RMSE:.2f} μg/m³。月份指标与目标浓度和事件强度共同变化，应用中应持续跟踪季节与过程转折。"),
        f('14_error_structure.png',212),p('图15　左：测试年逐月误差；右：按次日相对当日日均PM2.5的真实变动分组。变化阈值50 μg/m³用于事后误差诊断。','small'),
        t([['变化组','样本数','Ridge RMSE','持续性RMSE','Ridge Bias'],
            *[[{'Change <50':'绝对变动<50','Decrease >=50':'下降≥50','Increase >=50':'上升≥50'}[row.group],int(row.n),
               f'{row.RMSE:.2f}',f'{row.persistence_RMSE:.2f}',f'{row.Bias:.2f}'] for row in changes.itertuples()]], [120,64,108,108,103]),
        Spacer(1,7),p('浓度绝对变动小于50时，持续性RMSE为25.15，低于岭回归的29.44；快速下降时岭回归将误差由129.32降至60.46，快速上升时由94.74降至77.49 μg/m³。因此，总体改进主要体现在较大变化场景，平稳日保留简单基线仍有比较价值。'),
        p('岭回归在快速上升组平均低估58.94，在快速下降组平均高估37.86 μg/m³，显示对转折幅度的压缩。分组使用次日真实值，属于评估后的解释变量；设计能够提前切换模型的策略，需要另外训练和验证变化识别器。'),
    ])

if __name__=='__main__':main()
