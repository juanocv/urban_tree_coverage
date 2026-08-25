> 🇬🇧 **English:** [Read this page in English](../../README.md)

# Urban Tree Coverage

O Urban Tree Coverage estima a **cobertura arbórea visível** de vias urbanas a
partir de imagens do Google Street View, usando segmentação semântica e
panóptica. O pacote de produção fica em `urban_canopy/`; os checkouts de modelos
de terceiros (OneFormer via HuggingFace, Detectron2, DeepLab) ficam fora dos
limites deste pacote.

O indicador principal é contínuo e por imagem:

```
tree_coverage_ratio = pixels de árvore / todos os pixels da imagem   (em [0, 1])
tree_coverage_pct   = 100 * tree_coverage_ratio
```

O indicador `vegetation_coverage_ratio`, mais amplo, é reportado separadamente quando o
modelo consegue distingui-lo. Classes de árvore, grama e arbusto **nunca são
fundidas silenciosamente** — o mapeamento das classes do modelo para esses grupos
é explícito, inspecionável e sobrescrevível (`urban_canopy/models/taxonomy.py`).
Nenhuma faixa qualitativa ("pouca / média / muita vegetação") é produzida, e o
projeto mede área, nunca contagem. O [FAQ](faq.md#o-indicador) traz o raciocínio
por trás das duas decisões.

## O que faz?

1. **Aquisição** — Street View (com cache, registrando id do panorama e data de
   captura) ou imagens locais.
2. **Estratégia de vista** — vista única, ou um plano multi-vista determinístico,
   escolhido por configuração e nunca pela saída da segmentação.
3. **Segmentação** — quatro backends por trás de um contrato comum.
4. **Refinamento** — limpeza opcional e conservadora da máscara, com uma trava de
   crescimento que limita o quanto qualquer ajuste pode inflá-la.
5. **Indicadores** — razões de cobertura por imagem, com flags de qualidade e
   proveniência da captura.
6. **Agregação** — média / mediana / IQR / p25 / p75 entre as vistas de um local.
7. **Avaliação** — dois níveis independentes contra ground truth COCO manual:
   pixels (IoU, Dice/F1, precisão, revocação) e o próprio indicador de cobertura
   (MAE, RMSE, viés em pontos percentuais).
8. **Artefatos de auditoria** — por vista: RGB, máscaras bruta e refinada,
   overlays, JSON de métricas; além de exportações CSV/JSON por execução.

### O que cada backend oferece?

| Backend | Pré-treino | Classe de árvore |
|---|---|---|
| OneFormer | ADE20K-150 | `tree` (stuff) + `palm` |
| Mask2Former | ADE20K / COCO / Cityscapes | depende do checkpoint |
| Detectron2 panoptic FPN | COCO-panoptic 133 | `tree-merged` (stuff) |
| DeepLab V3+ | Cityscapes-19 | nenhuma (`vegetation` funde árvores+arbustos) |

Um backend cujo espaço de classes não consegue discriminar a classe árvore reporta
**nenhuma razão de cobertura**, em vez de rotular novamente o número de
vegetação. Veja [qual backend usar](faq.md#escolha-do-backend) para a comparação
medida.

## Estrutura do repositório

```text
urban_canopy/              Pacote Python usado pela CLI e pela API
urban_canopy/core/         Orquestração do pipeline, config, resultados, planos de vista
urban_canopy/io/           I/O de Street View, imagem e geoespacial, artefatos
urban_canopy/models/       Adaptadores de backend, taxonomia, factory
urban_canopy/processing/   Cobertura, refinamento, agregação multi-vista
urban_canopy/evaluation/   Ground truth COCO, métricas, intercâmbio de predições
urban_canopy/tests/        Testes unitários offline, somente CPU
docs/                      Arquitetura, protocolo de anotação, avaliação, FAQ
notebooks/                 Dois exemplos completos, executáveis sem chave de API
samples/images/            Pequeno conjunto curado de imagens para experimentar
samples/annotations/       Ground truth COCO manual dessas imagens
```

## Experimentando sem chave de API

`samples/images/` contém sete quadros curados — incluindo um caso negativo sem
árvores e uma varredura de quatro headings de um mesmo local — todos anotados
manualmente, cobrindo de 0% a 29% de cobertura arbórea rotulada.

```bash
python -m pip install -e ".[ml,notebooks]"
jupyter lab notebooks/
```

- [`01_getting_started.ipynb`](../../notebooks/01_getting_started.ipynb) — uma
  imagem de ponta a ponta: o indicador, a separação árvore/vegetação, máscaras
  bruta vs refinada e a trava de crescimento do refinamento.
- [`02_multiview_and_evaluation.ipynb`](../../notebooks/02_multiview_and_evaluation.ipynb)
  — agregação multi-vista e os dois níveis de avaliação.

## Instalação

Requer Python 3.10 ou superior no Windows ou Linux (a CI testa 3.10 e 3.13).

**Linux** — Debian/Ubuntu não distribuem o `venv` junto com o interpretador, e o
OpenCV faz link contra a libGL:

```bash
sudo apt install python3-venv libgl1 libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Ou use o script auxiliar:

```bash
./scripts/setup-dev.sh --api --ml                                          # Linux
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev.ps1 -WithApi -WithMl  # Windows
```

A instalação base já basta para os testes unitários e para importar o pacote: os
módulos adaptadores mantêm os imports de Torch, Transformers, Pillow, Torchvision
e Detectron2 no momento da construção. Rodar segmentação de verdade exige a
camada de ML:

```bash
python -m pip install -e ".[ml]"
```

O PyTorch em si fica por sua conta: instale a build de CPU ou CUDA compatível com
a sua máquina em [pytorch.org](https://pytorch.org/get-started/locally/).

Copie `.env.example` para `.env` e defina `GOOGLE_API_KEY` antes de chamadas ao
Street View. Importar módulos e rodar os testes unitários nunca precisa da chave.

A configuração específica de cada backend — o checkpoint e o pacote `network` do
DeepLab, a compilação do Detectron2 e sua falha de `pkg_resources`, tamanhos de
download, diagnóstico de CUDA — está no [`faq.md`](faq.md#instalação) e no
[`reproducibility.md`](reproducibility.md#configuração-por-backend).

## Executando

A instalação editável expõe o console script `tree-ai`
(`python -m urban_canopy.cli.main` é o mesmo ponto de entrada).

```bash
# Imagem local, vista única
tree-ai --image street.jpg --single-view --seg oneformer --device cpu

# Coordenadas, multi-vista (0/90/180/270 em torno de um heading de referência, por padrão)
tree-ai --lat -23.678479 --lon -46.559621 --multi-view --seg oneformer

# Endereço, multi-vista com o bearing da rua conhecido
tree-ai "Av. Paulista 1578, Sao Paulo" --multi-view --reference-heading 45 --offsets 90,270

# Tudo que uma avaliação ou auditoria precisa, no diretório desta execução
tree-ai --image street.jpg --save-artifacts
```

Avaliar contra o ground truth COCO do Roboflow, e conferir uma exportação antes
de rotular mais:

```bash
tree-ai evaluate --predictions artifacts_out/<run>/predictions.json \
                 --annotations annotations.json --report-json report.json

tree-ai validate-dataset --annotations annotations.json
```

Flags que vale conhecer de saída: `--no-refine` (baseline da máscara bruta),
`--allow-vegetation-proxy` (deixa a `vegetation` do Cityscapes substituir
árvores, registrando `tree_source="vegetation_proxy"`), `--view-mode` (planos
multi-vista determinísticos) e `--min-successful-views` (aborta uma execução que
produziu imagens de menos). `tree-ai --help` lista o resto; o
[FAQ](faq.md#execução-e-saídas) explica as que têm consequência.

### Onde os resultados vão parar

Cada invocação ganha seu próprio diretório sob `--outdir` (padrão
`artifacts_out/`), nomeado pelo timestamp e pelo backend:

```text
artifacts_out/
  20260818-104512_oneformer/
    run.json            manifesto, agregado, todas as vistas
    views.csv           uma linha por vista
    predictions.json    para o `tree-ai evaluate`
    views/
      000_street/       rgb.png  mask_raw.png  mask_refined.png
                        overlay_tree.png  metrics.json
      001_...           demais vistas, na ordem de aquisição
```

As execuções se acumulam em vez de sobrescrever, e nada é escrito a menos que uma
flag de saída peça.

## Web API

```bash
python -m pip install -e ".[api,ml]"
uvicorn urban_canopy.webapi:app --host 127.0.0.1 --port 8000
```

A API lê as mesmas configurações de backend que a CLI, a partir do `.env`.
`POST /analyse/single` e `POST /analyse/multi` retornam as métricas de cobertura
mais a proveniência de backend/checkpoint/taxonomia. `GET /ping` é uma sonda de
liveness; `GET /ready` confirma a inicialização do modelo e devolve a mesma
proveniência, incluindo um SHA-256 quando os pesos são locais. A documentação
interativa fica em `/docs`. A avaliação de datasets permanece na CLI.

Os dois endpoints de análise aceitam `return_overlays` (desligado por padrão),
que acrescenta PNGs em base64 do quadro RGB, da sobreposição de árvores e da
máscara refinada — no `/single` sob uma chave `overlays` no topo, no `/multi`
sob `overlays` em cada vista, permitindo comparar direções lado a lado. Elas
dominam o tamanho da resposta: um quadro 640x640 tem cerca de um megabyte de PNG
e cada vista carrega três, então o `/multi` recusa um plano maior que
`UC_API_MAX_OVERLAY_VIEWS` (padrão 8) em vez de servir uma resposta que ninguém
pediu para receber.

O `UC_API_TOKENS` (separado por vírgulas) liga a autenticação bearer: `/ready` e
os dois endpoints `/analyse` passam a exigir `Authorization: Bearer <token>`,
enquanto o `GET /ping` continua aberto para sondas de liveness. Vazio, a API fica
sem autenticação — tudo bem no localhost e imprudente em qualquer outro lugar,
porque ela chama uma API paga do Google a cada requisição: uma instância aberta
gasta sua cota para quem a encontrar. A inicialização registra qual modo está
ativo.

Nada aqui limita taxa: o semáforo de concorrência limita quantas inferências
rodam ao mesmo tempo, não quantas um portador de token pode fazer. Defina também
um teto de orçamento do lado do Google.

Um console web estático para esta API está em
[urban_canopy-web](https://github.com/juanocv/urban_canopy-web).

## Ground truth e avaliação

A rotulação acontece no Roboflow, exportada como **COCO Instance Segmentation**,
um polígono/máscara por árvore. O ground truth em nível de pixel é a união
desses polígonos.

Todo arquivo de predições embute um manifesto (versões de pacotes, nome do
modelo, dispositivo, taxonomia, configuração de refinamento, semente do RNG e
flags de runtime determinístico), de modo que qualquer número reportado pode ser
rastreado até a execução que o produziu.

- [`faq.md`](faq.md) — problemas de instalação, escolha de backend e por que as
  decisões de projeto são o que são
- [`annotation_protocol.md`](annotation_protocol.md) — o que conta como árvore,
  copas vs troncos, oclusões, árvores parciais, visibilidade mínima
- [`evaluation.md`](evaluation.md) — métricas, regras de casamento, convenções
  para casos vazios, política de divisão validação/teste
- [`architecture.md`](architecture.md) — contratos dos módulos e o mapeamento a
  partir dos componentes do `sidewalk_analysis`
- [`reproducibility.md`](reproducibility.md) — captura de ambiente e configuração
  específica de cada backend
- [`detectron2-windows.md`](detectron2-windows.md) — Detectron2 no Windows, e a
  questão do WSL

## Verificações de qualidade

```bash
./scripts/check.sh                                             # Linux
powershell -ExecutionPolicy Bypass -File .\scripts\check.ps1    # Windows
```

Ou individualmente:

```bash
python -m pytest --cov=urban_canopy --cov-report=term-missing \
  --cov-report=json:coverage.json --cov-fail-under=80
python -m ruff check urban_canopy
python -m black --check urban_canopy
python -m pyright
python scripts/check_coverage.py coverage.json --fail-under 60
```

A suíte padrão é offline e somente CPU; `pytest -m gpu` e `pytest -m network`
rodam as verificações excluídas. Veja o [FAQ](faq.md#desenvolvimento) para o que
cada gate cobra e por quê.

## Transparência sobre o uso de IA Generativa

Ferramentas de IA generativa foram utilizadas como apoio durante a concepção e o 
desenvolvimento deste projeto, incluindo atividades como discussão de alternativas 
de implementação, revisão e organização de código, elaboração de testes e revisão 
da documentação.

As sugestões e conteúdos produzidos com auxílio dessas ferramentas foram revisados, 
adaptados e validados pelo autor. As decisões de projeto, a implementação final, 
os experimentos, a interpretação dos resultados e a responsabilidade pelo conteúdo 
deste repositório permanecem integralmente sob responsabilidade do autor.

Esse uso de IA generativa como ferramenta de apoio ao desenvolvimento é distinto 
dos modelos de segmentação empregados pelo próprio Urban Tree Coverage como parte 
de seu pipeline de análise.

## Citação

```bibtex
@misc{urban_tree_coverage_2026,
  author = {Juan Oliveira de Carvalho},
  title = {Urban Tree Coverage: Visible Street-Level Tree Coverage from Street View Imagery Using Semantic Segmentation},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/juanocv/urban_tree_coverage}
}
```
