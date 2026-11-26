# Teste de funcionamento offline

**Resultado: APROVADO**

- Executado em: 26/11/2026 às 16:00:05
- Duração: 38 s
- Sistema: Windows 10
- Versão do sistema: EmotionScan v1.3
- Script: `scripts/testar_offline.py`

## Confirmação de que a rede estava desligada

Antes de qualquer verificação, o script tentou alcançar 5 alvos conhecidos (DNS da Cloudflare e do Google, GitHub, Hugging Face e PyPI).

**Nenhum respondeu — a rede estava efetivamente fora.**

## Verificações

- [x] Integridade dos modelos por SHA-256
- [x] Carregamento dos modelos ONNX
- [x] Inferência real sobre imagem de referência
- [x] Distribuição de probabilidades válida (soma 1,0)
- [x] Ausência de rosto não produz classificação
- [x] Suíte de testes automatizados
- [x] Abertura da interface gráfica

## Medições durante o teste

| Item | Valor |
|---|---|
| Carregamento dos modelos | 130,0 ms |
| Classe prevista na imagem de referência | Neutra |
| Confiança | 62,0% |
| Ciclo completo | 16,88 ms |
| Suíte de testes | 159 passed, 1 deselected in 35.20s |

## O que este teste comprova

A auditoria de código já mostrava **zero chamadas de rede** em `src/`. Isso é indício forte, mas não é prova: uma biblioteca poderia resolver DNS, verificar licença ou enviar telemetria sem aparecer numa busca por `import`.

Este teste fecha essa lacuna **executando o sistema com a rede fisicamente fora**.

## O que NÃO comprova

- Não cobre a **instalação** (`configurar.bat` precisa de internet uma   vez, para baixar os modelos).
- Não cobre outra máquina nem outro sistema operacional.
- Com `--sem-interface`, não cobre a janela gráfica.
