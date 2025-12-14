# infrastructure/terraform/main.tf
# Exemplo básico de Dashboard OCI via Terraform

variable "compartment_id" {}
variable "tenancy_ocid" {}

provider "oci" {
  tenancy_ocid = var.tenancy_ocid
  region       = "us-ashburn-1" # Ajuste sua região
}

resource "oci_management_dashboard_management_dashboards_import" "bot_dashboard" {
  # Nota: A API de Dashboards do OCI via Terraform é complexa (importação de JSON).
  # Abaixo, exemplo conceitual de Alarme, que é mais comum em TF.
}

resource "oci_monitoring_alarm" "high_latency" {
  compartment_id        = var.compartment_id
  destinations          = ["<OCID_DO_TOPIC_NOTIFICATION>"]
  display_name          = "Bot High Latency Alert"
  is_enabled            = true
  metric_compartment_id = var.compartment_id
  namespace             = "MarketBot_Prod"
  query                 = "TradeLagMs[1m].mean() > 5000"
  severity              = "CRITICAL"
  
  body                  = "Latência do Bot acima de 5s! Verifique logs."
  message_format        = "RAW"
  pending_duration      = "PT1M" # aguardar 1 min antes de disparar
}

resource "oci_monitoring_alarm" "missing_heartbeat" {
  compartment_id        = var.compartment_id
  destinations          = ["<OCID_DO_TOPIC_NOTIFICATION>"]
  display_name          = "Bot Down (Heartbeat Missing)"
  is_enabled            = true
  metric_compartment_id = var.compartment_id
  namespace             = "MarketBot_Prod"
  query                 = "BotHeartbeat[1m].absent()"
  severity              = "CRITICAL"
  
  body                  = "Bot parou de enviar Heartbeat!"
  pending_duration      = "PT1M"
}
